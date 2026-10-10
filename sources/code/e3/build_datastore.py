# -*- coding: utf-8 -*-
"""E3 建库：teacher-forcing 提取 (key=decoder 倒数第二层隐状态, value=下一 BPE token)。
库源 = 论文1 train_v2_strict.csv（置信度过滤后的 large-v3 伪标签池）确定性抽样。
用法：
  python build_datastore.py --hours 30 --out /root/autodl-tmp/e3_ds/pilot30h --device cuda:0
  python build_datastore.py --hours 0 --all --out /root/autodl-tmp/e3_ds/full --device cuda:0
输出：keys_shardXXXX.npy (fp16, L2 归一化), values_shardXXXX.npy (int32), meta.json, word_freq.json
"""
import argparse, csv, json, os, random, sys, time
from collections import Counter

import numpy as np
import soundfile as sf  # noqa: F401 保留给 info 检查
from faster_whisper.audio import decode_audio  # 统一音频加载：重采样 16k+下混
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "e1"))
from common import norm  # noqa: E402

MODEL_DIR = "/root/autodl-tmp/dev/kazakh_asr/Whisper/model/whisper-large-v3"
POOL_CSV = "/root/autodl-tmp/data/final_manifests/train_v2_strict.csv"
PROMPT = [50258, 50316, 50360, 50364]  # SOT, kk, transcribe, no_timestamps（probe_model.py 验证）
VOCAB = 51865
TS_IDS = set(range(50365, 51866))      # 时间戳区（probe 实测 [50365,51865]）

ap = argparse.ArgumentParser()
ap.add_argument("--pool-csv", default=POOL_CSV, help="池 manifest（默认=strict 586h；全量消融用 q2 metadata csv）")
ap.add_argument("--lang", default="kk", help="建库语言（改变 tokenizer 语言与 prompt 语言 token；tr/... ）")
ap.add_argument("--hours", type=float, default=30.0, help="抽样小时数；0 配合 --all 取全池")
ap.add_argument("--all", action="store_true")
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--shard", type=int, default=0, help="双卡分片：本进程处理 picked[shard::nshards]")
ap.add_argument("--nshards", type=int, default=1)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

proc = WhisperProcessor.from_pretrained(MODEL_DIR)
tok = proc.tokenizer
if args.lang != "kk":
    _p = tok.get_decoder_prompt_ids(language=args.lang, task="transcribe", no_timestamps=True)
    PROMPT = [PROMPT[0]] + [pid for _, pid in _p]
    print(f"prompt for {args.lang}: {PROMPT}", flush=True)
dev = args.device
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.float16).to(dev).eval()

# ---------- 抽样（确定性）----------
rows = list(csv.DictReader(open(args.pool_csv, encoding="utf-8")))
rng = random.Random(args.seed)
rng.shuffle(rows)
if args.all:
    picked = rows
else:
    picked, tot = [], 0.0
    for r in rows:
        picked.append(r)
        tot += float(r["duration"])
        if tot >= args.hours * 3600:
            break
picked.sort(key=lambda r: float(r["duration"]))  # 短句前排，减少 padding 浪费
picked = picked[args.shard::args.nshards]
print(f"picked {len(picked)} utts (shard {args.shard}/{args.nshards}), "
      f"{sum(float(r['duration']) for r in picked)/3600:.1f} h", flush=True)

# 全池词频（未配对文本代理，供分桶与低频子库用；与抽样无关，取整池一次）
freq_path = os.path.join(args.out, "word_freq.json")
if not os.path.exists(freq_path):
    cnt = Counter()
    for r in rows:
        cnt.update(norm(r.get("sentence") or r.get("wrd", ""), lang=args.lang).split())
    json.dump(cnt, open(freq_path, "w", encoding="utf-8"))
    print(f"word_freq.json: {len(cnt)} word types", flush=True)
del rows

# ---------- value 合法性：真 token 且非时间戳/特殊 ----------
def valid_value(i: int) -> bool:
    return 0 <= i < VOCAB and i not in TS_IDS

# ---------- 批量 teacher-forcing ----------
EOT = tok.eos_token_id
n_shard, n_entry = 0, 0
shard_keys, shard_vals, shard_cap = [], [], 200_000
t0, n_done = time.time(), 0


def flush_shard():
    global n_shard, n_entry, shard_keys, shard_vals
    if not shard_keys:
        return
    k = torch.cat(shard_keys).cpu().numpy().astype(np.float16)
    v = torch.cat(shard_vals).cpu().numpy().astype(np.int32)
    assert k.shape == (len(v), 1280) and np.isfinite(k).all()
    assert ((v >= 0) & (v < 50365)).all(), "Invalid datastore value"
    assert np.allclose(np.linalg.norm(k.astype(np.float32), axis=1), 1, atol=0.003)
    tag = args.shard * 1000 + n_shard
    np.save(os.path.join(args.out, f"keys_shard{tag:04d}.npy"), k)
    np.save(os.path.join(args.out, f"vals_shard{tag:04d}.npy"), v)
    n_entry += len(k)
    print(f"  shard {n_shard:04d}: {len(k)} entries (total {n_entry})", flush=True)
    n_shard += 1
    shard_keys, shard_vals = [], []


with torch.no_grad():
    for b0 in range(0, len(picked), args.batch):
        chunk = picked[b0:b0 + args.batch]
        feats, labels, labels_nopad = [], [], []
        for r in chunk:
            audio = decode_audio(r["file_path"], sampling_rate=16000)
            f = proc.feature_extractor(audio, sampling_rate=16000,
                                       padding="max_length", max_length=480000,
                                       return_tensors="np")["input_features"][0]
            feats.append(f)
            ids = tok.encode(norm(r.get("sentence") or r.get("wrd", ""), lang=args.lang), add_special_tokens=False)[:400]
            labels_nopad.append(ids)
            labels.append(PROMPT + ids + [EOT])
        maxlen = max(len(x) for x in labels)
        lab_mat = torch.full((len(chunk), maxlen), EOT, dtype=torch.long)
        att_mat = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for j, x in enumerate(labels):
            lab_mat[j, :len(x)] = torch.tensor(x)
            att_mat[j, :len(x)] = 1
        feats_t = torch.tensor(np.stack(feats), dtype=torch.float16).to(dev)
        enc = model.model.encoder(feats_t).last_hidden_state
        dec = model.model.decoder(lab_mat.to(dev), attention_mask=att_mat.to(dev),
                                  encoder_hidden_states=enc, output_hidden_states=True)
        pen = dec.hidden_states[-2]          # (B, L, 1280) 倒数第二层输出
        for j, ids in enumerate(labels_nopad):
            n = len(ids)
            # 位置 p 消费 labels[0..p]，预测 labels[p+1]；只取 p 使 value=labels[p+1] 为真 token
            # p ∈ [len(PROMPT)-1, len(PROMPT)+n-2]
            p0, p1 = len(PROMPT) - 1, len(PROMPT) + n - 1
            ks = pen[j, p0:p1]               # (n,) key
            vs = lab_mat[j, p0 + 1:p1 + 1]   # (n,) value = 下一 token
            assert torch.equal(vs, torch.tensor(ids, dtype=vs.dtype)), "key/value label shift"
            ks = torch.nn.functional.normalize(ks.float(), dim=-1).half()
            shard_keys.append(ks)
            shard_vals.append(vs)
            if sum(k.shape[0] for k in shard_keys) > shard_cap:
                flush_shard()
        n_done += len(chunk)
        if (b0 // args.batch) % 50 == 0:
            el = time.time() - t0
            print(f"  {n_done}/{len(picked)} utts  {el/60:.1f} min  "
                  f"RTF={el / max(1e-9, sum(float(r['duration']) for r in picked[:n_done])):.3f}",
                  flush=True)
flush_shard()

meta = {"hours": (sum(float(r["duration"]) for r in picked) / 3600) if picked else 0,
        "utts": len(picked), "entries": n_entry, "seed": args.seed, "shard": args.shard,
        "nshards": args.nshards,
        "pool": args.pool_csv, "model": MODEL_DIR, "prompt": PROMPT,
        "key_layer": "hidden_states[-2] L2-normalized fp16"}
mname = "meta.json" if args.nshards == 1 else f"meta_s{args.shard}.json"
from provenance import run_metadata, atomic_json
meta["provenance"] = run_metadata(args, inputs=[args.pool_csv, MODEL_DIR + "/config.json"])
atomic_json(os.path.join(args.out, mname), meta)
print("META:", json.dumps(meta), flush=True)
