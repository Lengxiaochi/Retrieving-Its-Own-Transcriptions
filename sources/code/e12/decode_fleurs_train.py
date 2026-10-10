# -*- coding: utf-8 -*-
"""FLEURS train 子集的 teacher 伪标签解码（2026-09-22 建，extras ③复验零标注同构修复）。
与 decode_cv_train.py 同款路径与输出契约（file_path, wrd=norm(hyp), duration, avg_logprob,
no_speech_prob）——下游 filter_manifest/build_datastore 无需任何改动。
FLEURS tsv 无表头 7 列（id, filename, raw, normed, chars, frames, gender）且无时长列：
时长用 soundfile 读音频头；filename 去重。用法：
  python decode_fleurs_train.py --lang sw_ke --out pseudo.csv [--limit N]"""
import argparse, csv, json, os, sys, time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "e1"))
from common import norm, NORMALIZATION_VERSION  # noqa: E402

FL_ROOT = "/root/autodl-tmp/share/fleurs"

ap = argparse.ArgumentParser()
ap.add_argument("--lang", required=True, help="FLEURS 语言码（如 sw_ke/ta_in/uz_uz）")
ap.add_argument("--split", default="train")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

import soundfile as sf
from faster_whisper import WhisperModel

short = args.lang.split("_")[0]          # fw.transcribe 语言提示用短码（sw/ta/uz）
base = f"{FL_ROOT}/{args.lang}"
rows, seen = [], set()
with open(f"{base}/trans/{args.split}.tsv", encoding="utf-8") as f:
    for line in f:
        parts = line.rstrip("\n").split("\t")
        if len(parts) != 7 or parts[1] in seen:
            raise ValueError("Malformed or duplicate FLEURS row")
        seen.add(parts[1])
        rows.append(parts[1])
if args.limit:
    rows = rows[:args.limit]
print(f"{args.lang}/{args.split}: {len(rows)} utts", flush=True)

fw = WhisperModel("/root/autodl-tmp/dev/kazakh_asr/Whisper/model/whisper-large-v3-ct2",
                  device="cuda", compute_type="float16")

from provenance import run_metadata, atomic_json
import ctranslate2
ctranslate2.set_random_seed(args.seed)
atomic_json(args.out + ".run.json", run_metadata(args, inputs=[f"{base}/trans/{args.split}.tsv"]))
out_rows, t0, done = [], time.time(), 0
for i, fid in enumerate(rows):
    path = f"{base}/audio/{args.split}/{fid}"
    try:
        dur = sf.info(path).duration
        segs, info = fw.transcribe(path, language=short, beam_size=1, vad_filter=False)
        segs = list(segs)   # faster-whisper 返回生成器，必须先物化
        raw = " ".join(s.text for s in segs)
        hyp = norm(raw, lang=short)
        alp = (sum(s.avg_logprob for s in segs) / max(1, len(segs))) if segs else -9.9
        nsp = min((s.no_speech_prob for s in segs), default=1.0)
    except Exception as e:
        print(f"  !! {fid}: {e}", flush=True)
        raise
    out_rows.append({"file_path": path, "wrd": hyp, "raw_hyp": raw,
                     "token_ids": json.dumps([t for s in segs for t in s.tokens]),
                     "normalization_version": NORMALIZATION_VERSION,
                     "duration": round(dur, 3),
                     "avg_logprob": round(alp, 4), "no_speech_prob": round(nsp, 4)})
    done += 1
    if done % 500 == 0:
        el = time.time() - t0
        print(f"  {done}/{len(rows)}  {el/3600:.2f}h  ETA {(el/done)*(len(rows)-done)/3600:.2f}h",
              flush=True)

assert len(out_rows) == len(rows) and rows, "Incomplete FLEURS pseudolabel output"
with open(args.out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0]))
    w.writeheader()
    w.writerows(out_rows)
print("DONE", args.out, len(out_rows), flush=True)
