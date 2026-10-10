# -*- coding: utf-8 -*-
"""D+2 T1：随机 value 库构造——keys 原样、values 全局随机置换。
用途：排除"kNN 增益来自乱码正则化而非真记忆"（kNN-Whisper 对照范式）。
输出：与源库同构的 datastore 目录（keys_shard*.npy 原样链接/复制 + vals 置换 + meta）。"""
import argparse, glob, json, os, shutil

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="源 datastore 目录")
ap.add_argument("--out", required=True)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
rng = np.random.RandomState(args.seed)

# 1) 收集全部 value（分 shard 读，逐 shard 置换——保证每条 key 得到某条随机 value）
n_total, shard_files = 0, sorted(glob.glob(os.path.join(args.src, "vals_shard*.npy")))
assert shard_files, f"源库无 vals_shard*: {args.src}"
for vf in shard_files:
    vals = np.load(vf)
    perm = rng.permutation(len(vals))
    np.save(os.path.join(args.out, os.path.basename(vf)), vals[perm].astype(np.int32))
    # keys 原样复制（fp16 快照）
    kf = vf.replace("vals_", "keys_")
    shutil.copyfile(kf, os.path.join(args.out, os.path.basename(kf)))
    n_total += len(vals)

# 2) vals_all 一致性：若源库有 vals_all.npy，按同一全局置换重建（与 shard 视图等价）
allf = os.path.join(args.src, "vals_all.npy")
if os.path.exists(allf):
    vals = np.load(allf, mmap_mode="r")
    perm = rng.permutation(len(vals))
    np.save(os.path.join(args.out, "vals_all.npy"), np.ascontiguousarray(vals[perm]).astype(np.int32))

# 3) meta 标记
meta_path = os.path.join(args.src, "meta.json")
meta = {}
if os.path.exists(meta_path):
    meta = json.load(open(meta_path))
meta["random_value_ablation"] = True
meta["seed"] = args.seed
meta["n_entries"] = n_total
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"))
print(f"random-value ds: {n_total} entries -> {args.out}")
