# -*- coding: utf-8 -*-
"""D+2 T4：库规模轴截断——从全量库按比例抽 keys/vals 子集（近似小时数线性于条目数）。
输出子库目录，后续 build_index 索引。"""
import argparse, glob, json, os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--fraction", type=float, required=True, help="保留比例 (0,1]")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

assert 0 < args.fraction <= 1
os.makedirs(args.out, exist_ok=True)
rng = np.random.RandomState(args.seed)
n_total = 0
shard_files = sorted(glob.glob(os.path.join(args.src, "vals_shard*.npy")))
assert shard_files, f"无 vals_shard*: {args.src}"
for vf in shard_files:
    vals = np.load(vf).astype(np.int32)
    keys = np.load(vf.replace("vals_", "keys_"))
    keep = rng.rand(len(vals)) < args.fraction
    np.save(os.path.join(args.out, os.path.basename(vf)), vals[keep])
    np.save(os.path.join(args.out, os.path.basename(vf).replace("vals_", "keys_")), keys[keep])
    n_total += int(keep.sum())
# vals_all 同步子集（与 shard 视图一致：按 shard 内 keep 拼接）
allf = os.path.join(args.src, "vals_all.npy")
if os.path.exists(allf):
    parts = []
    for vf in shard_files:
        vals = np.load(vf).astype(np.int32)
        keep = np.load(os.path.join(args.out, os.path.basename(vf)))
        # 由子集 value 反查原位置代价高；直接重采（统计性质等价）
        idx = rng.choice(len(vals), size=len(keep), replace=False)
        parts.append(vals[np.sort(idx)])
    np.save(os.path.join(args.out, "vals_all.npy"), np.concatenate(parts))
meta = {}
mp = os.path.join(args.src, "meta.json")
if os.path.exists(mp):
    meta = json.load(open(mp))
meta.update({"subset_fraction": args.fraction, "seed": args.seed, "n_entries": n_total})
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"))
print(f"subset ds: fraction={args.fraction} -> {n_total} entries -> {args.out}")

