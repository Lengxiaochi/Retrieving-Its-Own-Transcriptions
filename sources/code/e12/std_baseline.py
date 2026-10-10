# -*- coding: utf-8 -*-
"""标准 faster-whisper 解码基线（审阅 §3.6 / 排期必做：与检索系统同数据、同归一化、同硬件对照）。
用法：python std_baseline.py --beam 1 --out results/e1std_greedy_test.json [--limit N]"""
import argparse, json, os, sys, time

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "e1"))
from common import norm, load_test, load_dev  # noqa: E402
import jiwer  # noqa: E402
from faster_whisper import WhisperModel  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--beam", type=int, default=1)
ap.add_argument("--split", choices=["dev", "test"], default="test")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

data = (load_dev if args.split == "dev" else load_test)(limit=args.limit)
print(f"KSC2 {args.split}: {len(data)} utts, beam={args.beam}", flush=True)
fw = WhisperModel("/root/autodl-tmp/dev/kazakh_asr/Whisper/model/whisper-large-v3-ct2",
                  device="cuda", compute_type="float16")
per, t0 = {}, time.time()
for i, (uid, wav, ref) in enumerate(data):
    segs, info = fw.transcribe(wav, language="kk", beam_size=args.beam, vad_filter=False)
    hyp = norm(" ".join(s.text for s in segs))
    per[uid] = {"hyp": hyp}
    if (i + 1) % 500 == 0:
        print(f"  {i+1}/{len(data)}  {(time.time()-t0)/60:.1f}min", flush=True)
e = d = 0
for uid, _, ref in data:
    if uid not in per: continue
    m = jiwer.process_words(ref, per[uid]["hyp"])
    e += m.substitutions + m.deletions + m.insertions
    d += m.hits + m.substitutions + m.deletions
summary = {"model": f"standard faster-whisper large-v3 beam={args.beam}",
           "n": len(per), "WER": round(100 * e / max(1, d), 2),
           "wall_min": round((time.time() - t0) / 60, 1)}
json.dump({"summary": summary, "per_utt": per}, open(args.out, "w"), ensure_ascii=False)
json.dump(summary, open(args.out.replace(".json", "_summary.json"), "w"), indent=1)
print("SUMMARY:", summary, flush=True)
