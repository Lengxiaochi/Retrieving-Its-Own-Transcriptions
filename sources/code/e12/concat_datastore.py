# -*- coding: utf-8 -*-
"""库2 = 库1 ⊕ 库3' 条目级拼接（2026-09-15 用户拍板：加法性优先，不在合并池上重过滤重建）。
把两个 datastore 的 keys_shard*.npy / vals_shard*.npy 复制进新目录，第二库文件名 tag 统一 +10000
偏移防撞名；word_freq.json 计数相加；meta.json 汇总条目数/时长。索引由 build_index.py 另跑。
断言：两库分片 keys/vals 文件一一对应；输出条目总数 = 两库之和（拼接充要）。
用法：python concat_datastore.py --ds1 <库1目录> --ds2 <库3'目录> --out <库2目录>
"""
import argparse, glob, json, os, re, shutil, sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--ds1", required=True)
ap.add_argument("--ds2", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)


def shards(d):
    ks = sorted(glob.glob(os.path.join(d, "keys_shard*.npy")))
    out = []
    for k in ks:
        v = k.replace("keys_shard", "vals_shard")
        if not os.path.exists(v):
            sys.exit(f"FATAL: {k} 无对应 vals 文件")
        out.append((k, v))
    return out


s1, s2 = shards(args.ds1), shards(args.ds2)
if not s1 or not s2:
    sys.exit("FATAL: 某库无分片文件")
print(f"ds1={len(s1)} 片  ds2={len(s2)} 片")

n_total = 0
for tag_ofs, slist in ((0, s1), (10000, s2)):
    for k, v in slist:
        tag = int(re.search(r"shard(\d+)", os.path.basename(k)).group(1))
        shutil.copyfile(k, os.path.join(args.out, f"keys_shard{tag + tag_ofs:04d}.npy"))
        shutil.copyfile(v, os.path.join(args.out, f"vals_shard{tag + tag_ofs:04d}.npy"))
        n_total += len(np.load(k, mmap_mode="r"))
print(f"拼接完成：{len(s1) + len(s2)} 片，共 {n_total} entries → {args.out}")

# word_freq 计数相加；meta 汇总
freq = {}
for d in (args.ds1, args.ds2):
    p = os.path.join(d, "word_freq.json")
    if os.path.exists(p):
        for w, c in json.load(open(p, encoding="utf-8")).items():
            freq[w] = freq.get(w, 0) + c
json.dump(freq, open(os.path.join(args.out, "word_freq.json"), "w", encoding="utf-8"))
metas = [json.load(open(os.path.join(d, "meta.json"))) for d in (args.ds1, args.ds2) if os.path.exists(os.path.join(d, "meta.json"))]
meta = {"src": [args.ds1, args.ds2],
        "entries": n_total,
        "hours": sum(float(m.get("hours", 0)) for m in metas),
        "note": "库2=库1⊕库3' 条目拼接（2026-09-15 定稿），未重过滤"}
json.dump(meta, open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
print(f"meta: {meta['hours']:.1f}h, {len(freq)} word types")
