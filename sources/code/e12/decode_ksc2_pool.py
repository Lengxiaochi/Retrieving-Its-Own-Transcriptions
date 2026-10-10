# -*- coding: utf-8 -*-
"""KSC2 池伪标签解码（f-w 标准路径，v1 定版 2026-09-14：HF 批量连败后用户拍板回归 f-w）。
与 decode_cv_train 同协议（kk prompt/beam1/无 VAD），输出格式兼容 filter_manifest.py。
短句多导致实时率 ~14x：单 shard ~53h，双卡并行 ~2.5 天——用户知情拍板。
数据指纹：日志首行自证 manifest/条数/时长（坑 20 防线）。"""
import argparse, csv, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", required=True)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--num-shards", type=int, default=1)
ap.add_argument("--out", required=True)
args = ap.parse_args()

csv.field_size_limit(sys.maxsize)
rows = list(csv.DictReader(open(args.manifest, encoding="utf-8"), quoting=csv.QUOTE_NONE))
if args.limit:
    rows = rows[: args.limit]
rows = rows[args.shard:: args.num_shards]
tot_h = sum(float(r["duration"]) for r in rows) / 3600
print(f"数据指纹: manifest={os.path.basename(args.manifest)} shard={args.shard}/{args.num_shards} "
      f"n={len(rows)} 总时长={tot_h:.1f}h 首wav={os.path.basename(rows[0]['file_path'])}", flush=True)

from faster_whisper import WhisperModel
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e1"))
from common import norm  # noqa: E402

fw = WhisperModel("/root/autodl-tmp/dev/kazakh_asr/Whisper/model/whisper-large-v3-ct2",
                  device="cuda", compute_type="float16")

out_rows, t0, done = [], time.time(), 0
for r in rows:
    path = r["file_path"]
    try:
        segs, info = fw.transcribe(path, language="kk", beam_size=1, vad_filter=False)
        segs = list(segs)   # faster-whisper 返回生成器，必须先物化
        hyp = norm(" ".join(s.text for s in segs))
        alp = (sum(s.avg_logprob for s in segs) / max(1, len(segs))) if segs else -9.9
        nsp = min((s.no_speech_prob for s in segs), default=1.0)
    except Exception as e:
        print(f"  !! {os.path.basename(path)}: {type(e).__name__} {e}", flush=True)
        continue
    out_rows.append({"file_path": path, "wrd": hyp,
                     "duration": round(float(r["duration"]), 3),
                     "avg_logprob": round(alp, 4), "no_speech_prob": round(nsp, 4)})
    done += 1
    if done % 500 == 0:
        el = time.time() - t0
        audio_h = sum(float(x["duration"]) for x in out_rows) / 3600
        print(f"  {done}/{len(rows)}  {el/3600:.2f}h  实时率 {audio_h/max(el/3600,1e-9):.1f}x  "
              f"ETA {(el/done)*(len(rows)-done)/3600:.2f}h", flush=True)

with open(args.out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["file_path", "wrd", "duration", "avg_logprob", "no_speech_prob"])
    w.writeheader()
    w.writerows(out_rows)
print(f"DONE {args.out} {len(out_rows)} 条", flush=True)
