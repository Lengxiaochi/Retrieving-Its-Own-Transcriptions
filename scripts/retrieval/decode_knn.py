# -*- coding: utf-8 -*-
"""E3 插值解码：手写 greedy（KV cache）+ 逐步 kNN 检索，p = λ·p_kNN + (1−λ)·p_model。
λ=0 / 不给 --ds 时 = 空库基线（同代码路径）。
用法示例：
  # 基线
  python decode_knn.py --split dev --limit 1000 --out ../../results/e3/dev1k_lambda0.json
  # 插值
  python decode_knn.py --split dev --limit 1000 --ds /root/autodl-tmp/e3_ds/pilot30h \
      --lambda 0.3 --k 4 --T 100 --out ../../results/e3/dev1k_lam03.json
"""
import argparse, glob, json, os, sys, time

import numpy as np
import soundfile as sf  # noqa: F401 保留给 info 检查
from faster_whisper.audio import decode_audio  # 统一音频加载：重采样 16k+下混
import torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "shared"))
from common import load_dev, load_test, norm, score_and_save, compression_ratio  # noqa: E402

MODEL_DIR = "/root/autodl-tmp/dev/kazakh_asr/Whisper/model/whisper-large-v3"
PROMPT = [50258, 50316, 50360, 50364]
VOCAB, EOT = 51866, 50257   # large-v3 logits 宽度实测 51866；时间戳 [50365,51865] 共 1501 个
TS_IDS = np.arange(50365, 51866)

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["dev", "test", "train"], default="dev")
ap.add_argument("--limit", type=int, default=1000)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--ds", default=None, help="datastore 目录；缺省=空库基线")
ap.add_argument("--index", default="", help="预建 faiss 索引路径（全量库 IVFPQ）；空=从 --ds 分片组 Flat")
ap.add_argument("--nprobe", type=int, default=32)
ap.add_argument("--lambda", dest="lam", type=float, default=0.0)
ap.add_argument("--k", type=int, default=4)
ap.add_argument("--T", type=float, default=100.0, help="kNN 核温度（Whisper-kNN 默认 100）")
ap.add_argument("--search-mult", type=int, default=1, help="检索 top-(k*mult) 再截断")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--lang", default="kk", help="解码语言代码（kk/tr/...）；改变 prompt 的语言 token")
ap.add_argument("--fleurs-lang", default=None,
                help="FLEURS 评测（如 kk_kz/ky_kg/tr_tr/uz_uz）；忽略 --split 数据源")
ap.add_argument("--cv-lang", default=None,
                help="设置后从 Common Voice 加载数据（--cv-lang 即语言代码），忽略 --split 的 KSC2 数据源")
ap.add_argument("--gate-conf", type=float, default=0.0,
                help="置信门控（FLARE 式）：p_model 最大概率 ≥ 该值时本步 λ=0；0=关闭")
ap.add_argument("--cr-anneal", type=float, default=0.0,
                help="CR 退火：生成文本压缩比 > 该值后该句 λ 永久置 0；0=关闭")
ap.add_argument("--cr-start", type=int, default=8, help="CR 退火开始监测的最少已生成 token 数")
ap.add_argument("--retry-cr", type=float, default=0.0,
                help="终检重试（Whisper 检测→重试的 λ 版）：输出 CR > 该值的句子用 --retry-lam 重解；0=关闭")
ap.add_argument("--retry-lam", type=float, default=0.0, help="重试时使用的 λ（通常 0=撤掉记忆）")
ap.add_argument("--fallback-thr", type=float, default=0.0,
                help="温度回退（f-w 标准）：输出 CR > 该值的句子用采样温度重解；0=关闭")
ap.add_argument("--fallback-temp", type=float, default=0.4, help="回退采样温度（第二档 = 2×）")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

from transformers import WhisperForConditionalGeneration, WhisperProcessor  # noqa: E402

proc = WhisperProcessor.from_pretrained(MODEL_DIR)
tok = proc.tokenizer
model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_DIR, dtype=torch.float16).to(args.device).eval()

# ---------- datastore ----------
knn = None
if args.ds and args.lam > 0:
    import faiss
    vals_path = os.path.join(args.ds, "vals_all.npy")
    if os.path.exists(args.index):                    # 预建 IVFPQ（全量库）
        cpu_idx = faiss.read_index(args.index)
        cpu_idx.nprobe = args.nprobe
        vals = np.load(vals_path, mmap_mode="r")
        # A40 smem 48KB < PQ64 查找表 64KB → GpuClonerOptions 无 useFloat16LookupTables，
        # 需走 GpuIndexIVFPQConfig 直接构造 + copyFrom
        cfg = faiss.GpuIndexIVFPQConfig()
        cfg.useFloat16LookupTables = True
        res = faiss.StandardGpuResources()
        devid = int(args.device.split(":")[1])
        cfg.device = devid
        gpu_idx = faiss.GpuIndexIVFPQ(res, cpu_idx.d, cpu_idx.nlist, cpu_idx.pq.M, 8,
                                      faiss.METRIC_L2, cfg)
        gpu_idx.copyFrom(cpu_idx)
        gpu_idx.nprobe = args.nprobe
        idx = gpu_idx
        print(f"datastore: IVFPQ index {cpu_idx.ntotal} entries, nprobe={args.nprobe}", flush=True)
    else:                                             # 试点：分片直接组 Flat
        kss, vss = [], []
        for kf in sorted(glob.glob(os.path.join(args.ds, "keys_shard*.npy"))):
            kss.append(np.load(kf))
            vss.append(np.load(kf.replace("keys_", "vals_")))
        keys = np.concatenate(kss).astype(np.float32)
        vals = np.concatenate(vss)
        cpu_idx = faiss.IndexFlatL2(keys.shape[1])
        cpu_idx.add(keys)
        idx = cpu_idx
        print(f"datastore: flat {len(vals)} entries from {args.ds}", flush=True)
        del kss, vss, keys
    assert cpu_idx.ntotal == len(vals), "index/value alignment mismatch"
    vals_t = torch.from_numpy(np.ascontiguousarray(vals)).to(args.device)
    knn = {"idx": idx, "vals": vals_t, "k": min(args.k * args.search_mult, len(vals))}
    del vals


# ---------- 数据 ----------
if args.fleurs_lang:
    sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "pseudo_labels"))
    from common_cv import load_fleurs
    data = load_fleurs(args.fleurs_lang, args.split)
elif args.cv_lang:
    # 坑 20（2026-09-14）：此分支曾在 09-13 FLEURS 编辑中被覆盖丢失，--cv-lang 静默回退 KSC2
    sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "pseudo_labels"))
    from common_cv import load_cv
    data = load_cv(args.cv_lang, "dev" if args.split == "dev" else "test")
else:
    data = (load_dev if args.split == "dev" else load_test)()
data = data[args.offset: args.offset + args.limit] if args.limit else data[args.offset:]
# prompt 语言自动对齐数据语言（D+2 阶段一 tr_random 漏 --lang tr 用 kk prompt 的教训）
# 坑 21（2026-09-23）：原对齐只写 --cv-lang 分支，--fleurs-lang 同样漏对齐——FLEURS 音频
# 全用 kk prompt 跑出西里尔伪哈语（sw λ0 WER 107.6 vs kk 正确配对 32.4，hyp 全西里尔实锤）；
# 已污染须重跑：xling ky/tr/uz λ0.5、sw/ta λ0 基线；kk_kz 点（kk 音频本就该 kk prompt）不受影响。
if args.cv_lang and args.lang == "kk":
    args.lang = args.cv_lang
if args.fleurs_lang and args.lang == "kk":
    args.lang = args.fleurs_lang.split("_")[0]
# 坑21续（2026-09-23）：ky 不在 Whisper tokenizer 语言表（get_decoder_prompt_ids 直接抛
# ValueError）——xling ky 点用最近支持语 kk 做 prompt 代理（吉/哈同属钦察突厥语支、同西里尔）；
# λ0 与 λ0.5 同用 kk prompt，配对内库效应对比依然干净；model 串如实记 lang=kk。
LANG_PROXY = {"ky": "kk"}
if args.lang in LANG_PROXY:
    print(f"[proxy] {args.lang} 不在 Whisper 语言表，prompt 改用最近支持语 {LANG_PROXY[args.lang]}")
    args.lang = LANG_PROXY[args.lang]
# 数据指纹：日志必须能自证加载的是哪个数据集（首uid/音频名/参考首句）
print(f"数据指纹: n={len(data)} 首uid={data[0][0]} 音频={os.path.basename(data[0][1])} ref[:30]={data[0][2][:30]}", flush=True)
print(f"split={args.split}{' cv=' + args.cv_lang if args.cv_lang else ''} "
      f"n={len(data)} λ={args.lam} k={args.k} T={args.T}", flush=True)

# prompt 语言 token 随 --lang 动态构造（kk 的硬编码值即 PROMPT 常量，等价）
if args.lang != "kk":
    _p = tok.get_decoder_prompt_ids(language=args.lang, task="transcribe", no_timestamps=True)
    PROMPT = [PROMPT[0]] + [pid for _, pid in _p]
    print(f"prompt for {args.lang}: {PROMPT}", flush=True)

# Record the chosen seed for this run; this does not establish seeds of archived runs.
torch.manual_seed(args.seed)
np.random.seed(args.seed)

TS_MASK = torch.zeros(VOCAB)
TS_MASK[TS_IDS] = float("-inf")   # 屏蔽时间戳：禁止输出，其余放行
TS_MASK = TS_MASK.to(args.device)

@torch.no_grad()
def decode_batch(items, lam_override=None, sampling_temp=0.0):
    """items: [(uid, wav, ref)]。返回 [(uid, hyp, lat_ms)]。
    lam_override: 本批有效 λ；sampling_temp: >0 按温度采样（温度回退用）。"""
    lam_eff = args.lam if lam_override is None else lam_override
    feats = []
    for _, wav, _ in items:
        audio = decode_audio(wav, sampling_rate=16000)
        feats.append(proc.feature_extractor(audio, sampling_rate=16000,
                                            padding="max_length", max_length=480000,
                                            return_tensors="np")["input_features"][0])
    x = torch.tensor(np.stack(feats), dtype=torch.float16).to(args.device)
    enc = model.model.encoder(x).last_hidden_state
    B = len(items)
    ids = torch.tensor([PROMPT] * B, device=args.device)
    past, done, out = None, [False] * B, [[] for _ in range(B)]
    cr_latched = [False] * B          # CR 退火：触发后该句 λ 永久置 0
    t0 = time.time()
    first = True
    while not all(done) and ids.shape[1] < 448:
        dec = model.model.decoder(ids if first else ids[:, -1:],
                                  encoder_hidden_states=enc, past_key_values=past,
                                  use_cache=True, output_hidden_states=True)
        past = dec.past_key_values
        last = dec.last_hidden_state[:, -1]                       # 最终层（post-LN）→ logits
        pen = dec.hidden_states[-2][:, -1]                        # 倒数第二层 → 检索 query
        logits = model.proj_out(last).float()
        logits = logits + TS_MASK                                  # 屏蔽时间戳
        p_model = torch.softmax(logits, -1)
        # 逐步 λ：置信门控 / CR 退火（默认恒定 lam_eff）
        lam_t = torch.full((B,), lam_eff, device=args.device)
        if knn is not None:
            if args.gate_conf > 0:
                conf = p_model.max(-1).values
                lam_t = torch.where(conf >= args.gate_conf,
                                    torch.zeros_like(lam_t), lam_t)
            if args.cr_anneal > 0:
                for j in range(B):
                    if not cr_latched[j] and len(out[j]) >= args.cr_start:
                        if compression_ratio(tok.decode(out[j])) > args.cr_anneal:
                            cr_latched[j] = True
                    if cr_latched[j]:
                        lam_t[j] = 0.0
        if knn is not None and bool((lam_t > 0).any()):
            q = torch.nn.functional.normalize(pen.float(), dim=-1)
            D, I = knn["idx"].search(q.contiguous().cpu().numpy(), knn["k"])
            D = torch.from_numpy(D).to(args.device).float()        # (B, k) squared-L2
            I = torch.from_numpy(I).to(args.device)
            vk = knn["vals"][I]                                    # (B, k)
            w = torch.exp(-D / args.T)
            pk = torch.zeros(B, VOCAB, device=args.device)
            pk.scatter_add_(1, vk, w)
            pk = pk / pk.sum(-1, keepdim=True).clamp_min(1e-9)
            p = (1 - lam_t).unsqueeze(-1) * p_model + lam_t.unsqueeze(-1) * pk
        else:
            p = p_model
        if sampling_temp > 0:
            ps_ = p.pow(1.0 / sampling_temp)
            ps_ = ps_ / ps_.sum(-1, keepdim=True).clamp_min(1e-9)
            nxt = torch.multinomial(ps_, 1).squeeze(-1)
        else:
            nxt = p.argmax(-1)
        for j in range(B):
            if done[j]:
                nxt[j] = EOT
            elif int(nxt[j]) == EOT:
                done[j] = True
            else:
                out[j].append(int(nxt[j]))
        ids = torch.cat([ids, nxt.unsqueeze(1)], 1)
        first = False
    lat = (time.time() - t0) * 1000 / B
    res = []
    for (uid, _, _), toks in zip(items, out):
        raw = tok.decode(toks, skip_special_tokens=True)
        hyp = norm(raw, lang=args.cv_lang or (args.fleurs_lang or args.lang).split("_")[0])
        res.append((uid, hyp, lat, raw, toks))
    return res


hyps, lats = [], {}
raw_hyps, token_ids = [], []
from provenance import run_metadata, atomic_json
metadata = run_metadata(args, data, [MODEL_DIR + "/config.json", args.index] if args.index else [MODEL_DIR + "/config.json"])
metadata["timing_scope"] = "wall time includes audio loading, feature extraction, encoder, decoder, and all retries; excludes model/index loading"
atomic_json(args.out + ".run.json", metadata)
n_retried = 0
t_start = time.time()
for b0 in range(0, len(data), args.batch):
    items = data[b0:b0 + args.batch]
    results = decode_batch(items)
    # 温度回退（f-w 标准）：CR 超阈值的句子用采样温度重解（升档两次）
    if args.fallback_thr > 0:
        need = [j for j, (_, hyp, _, _, _) in enumerate(results)
                if compression_ratio(hyp) > args.fallback_thr]
        for temp in (args.fallback_temp, min(1.0, args.fallback_temp * 2)):
            if not need:
                break
            redone = decode_batch([items[j] for j in need], sampling_temp=temp)
            still = []
            for j, r in zip(need, redone):
                if compression_ratio(r[1]) > args.fallback_thr:
                    still.append(j)
                else:
                    results[j] = r
            need = still
    if args.retry_cr > 0 and args.lam > 0:
        need = [i for i, (_, hyp, _, _, _) in enumerate(results)
                if compression_ratio(hyp) > args.retry_cr]
        if need:
            redone = decode_batch([items[i] for i in need], lam_override=args.retry_lam)
            for i, res in zip(need, redone):
                results[i] = res
            n_retried += len(need)
    for uid, hyp, lat, raw, toks in results:
        raw_hyps.append(raw)
        token_ids.append(toks)
        hyps.append(hyp)
        lats[uid] = lat
    if (b0 // args.batch) % 20 == 0:
        print(f"  {b0 + len(data[b0:b0+args.batch])}/{len(data)}  {(time.time()-t_start)/60:.1f} min",
              flush=True)

summary = score_and_save(data, hyps, args.out,
                         f"knn-greedy lang={args.lang} λ={args.lam} k={args.k} T={args.T} "
                         f"gate={args.gate_conf or '-'} crA={args.cr_anneal or '-'} "
                         f"retry={args.retry_cr or '-'}/λ{args.retry_lam} fb={args.fallback_thr or '-'} ds={args.ds}",
                         raw_hyps=raw_hyps, token_ids=token_ids, metadata=metadata)
summary["seed"] = args.seed
summary["n_retried"] = n_retried
summary["retry_rate_pct"] = round(100 * n_retried / max(1, len(data)), 2)
summary["mean_lat_ms_per_utt"] = round(sum(lats.values()) / len(lats), 1)
summary["wall_min"] = round((time.time() - t_start) / 60, 1)
artifact = json.load(open(args.out, encoding="utf-8"))
artifact["summary"] = summary
atomic_json(args.out, artifact)
atomic_json(args.out.replace(".json", "_summary.json"), {"summary": summary})
print("SUMMARY:", json.dumps(summary), flush=True)

