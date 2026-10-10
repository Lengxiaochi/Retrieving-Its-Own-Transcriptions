# -*- coding: utf-8 -*-
"""E12 伪标签置信过滤（论文1 同款阈值）：ALP 硬下限 -1.0 + 前 80% 分位、CR≤2.4、TPS≤20。
用法：python filter_manifest.py --in pseudo.csv --out filtered.csv"""
import argparse, csv, zlib

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--floor", type=float, default=-1.0)
ap.add_argument("--quantile", type=float, default=0.80)
ap.add_argument("--max-cr", type=float, default=2.4)
ap.add_argument("--max-tps", type=float, default=20.0)
args = ap.parse_args()

rows = list(csv.DictReader(open(args.input, encoding="utf-8")))


def cr(t):
    b = t.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


kept = []
for r in rows:
    if not r["wrd"].strip():
        continue
    r["cr"] = cr(r["wrd"])
    r["tps"] = len(r["wrd"]) / max(0.1, float(r["duration"]))
    if float(r["avg_logprob"]) > args.floor and r["cr"] <= args.max_cr and r["tps"] <= args.max_tps:
        kept.append(r)
kept.sort(key=lambda r: -float(r["avg_logprob"]))
kept = kept[:int(len(kept) * args.quantile)]
kept.sort(key=lambda r: r["file_path"])
if not kept:
    raise ValueError("No pseudolabels pass filtering")
with open(args.out, "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(kept[0].keys()))
    w.writeheader()
    w.writerows(kept)
print(f"filtered: {len(kept)}/{len(rows)} kept → {args.out}")
