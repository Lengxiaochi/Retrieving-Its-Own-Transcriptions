# -*- coding: utf-8 -*-
"""E3 全量库 IVFPQ 索引：从 datastore 分片训练 + 增量添加，落盘 faiss 索引。
配置照 Whisper-kNN：nlist=2048、PQ 64B（M=64, 8bit）、检索 nprobe=32、L2、key 已 L2 归一化。
用法：python build_index.py --ds /root/autodl-tmp/e3_ds/full586h --out /root/autodl-tmp/e3_ds/full586h.index"""
import argparse, glob, json, os

import faiss
import numpy as np

faiss.omp_set_num_threads(min(32, os.cpu_count() or 8))

ap = argparse.ArgumentParser()
ap.add_argument("--ds", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--nlist", type=int, default=2048)
ap.add_argument("--m", type=int, default=64, help="PQ 子量化器数（每子量化器 1 字节 → m 字节/向量）")
ap.add_argument("--nprobe", type=int, default=32)
ap.add_argument("--train-max", type=int, default=1_000_000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

shards = sorted(glob.glob(os.path.join(args.ds, "keys_shard*.npy")))
n_all = sum(np.load(k.replace("keys_", "vals_"), mmap_mode="r").shape[0] for k in shards)
print(f"{len(shards)} shards, {n_all} entries", flush=True)

d = 1280
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, args.nlist, args.m, 8)
index.nprobe = args.nprobe
index.verbose = True        # 训练阶段进度可见
index.cp.niter = 10         # 聚类 10 迭代（检索质量损失可忽略）
index.pq.cp.niter = 10
index.cp.seed = args.seed
index.pq.cp.seed = args.seed

# 训练样本：跨分片均匀抽样
rng = np.random.default_rng(args.seed)
per = args.train_max // len(shards)
tr_parts = []
for k in shards:
    n = np.load(k, mmap_mode="r").shape[0]
    take = rng.choice(n, min(per, n), replace=False)
    tr_parts.append(np.asarray(np.load(k, mmap_mode="r"))[take].astype(np.float32))
tr = np.concatenate(tr_parts)
del tr_parts
print(f"training on {len(tr)} vectors...", flush=True)
index.train(tr)
del tr

for i, kf in enumerate(shards):
    k = np.load(kf).astype(np.float32)
    assert k.shape == (len(np.load(kf.replace("keys_", "vals_"), mmap_mode="r")), d)
    assert np.isfinite(k).all()
    index.add(k)
    print(f"  added shard {i} ({index.ntotal})", flush=True)

vals = np.concatenate([np.load(k.replace("keys_", "vals_")) for k in shards])
assert len(vals) == index.ntotal == n_all
np.save(os.path.join(args.ds, "vals_all.npy"), vals.astype(np.int32))
faiss.write_index(index, args.out)
meta = {"seed": args.seed, "entries": int(index.ntotal), "nlist": args.nlist, "m": args.m,
        "nprobe": args.nprobe, "size_mb": round(os.path.getsize(args.out) / 1e6, 1)}
json.dump(meta, open(args.out + ".json", "w"))
print("META:", json.dumps(meta), flush=True)
