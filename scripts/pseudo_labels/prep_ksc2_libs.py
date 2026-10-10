# -*- coding: utf-8 -*-
"""KSC2/YouTube 库准备（CPU，双 shard 解码完成后、建库前）。
2026-09-15 晚按用户拍板重构：库5 取消；2×2（标签源×音频选择）全做；库2=库1+库3' 拼接。
模式：
  merge      shard0+shard1 → e12_ksc2_pool_full.csv（file_path 唯一断言；=库1f 全量伪标直接建库输入）
  join-gold  金标 manifest ∩ 库1过滤后音频 → ksc2_lib4_gold.csv（2×2 金标×过滤臂，同音频换标签源）
  q2-prep    q2 metadata 列适配（sentence→wrd，保留 alp/nsp/duration）→ yt_lib3p_input.csv（库3' 过滤输入）
置信过滤由 filter_manifest.py 单独执行（单一事实来源）：库1=pool_full 过滤、库3'=yt_lib3p_input 过滤；
库1f=pool_full 直接建库、库4f=ksc2_train_gold_v2.csv 直接建库（全量金标零空文本已验 09-15），无需本脚本。
断言：合并≈627,824（<1% 损失，解码跳过坏音频为合法少量）；file_path 全局唯一；
     join-gold 覆盖率=100%（pool/gold 同一次文件系统全扫必须 1:1）；q2-prep=628,002 行原样通过+音频抽查。
用法：
  python prep_ksc2_libs.py merge --shard0 ... --shard1 ... --outdir /root/autodl-tmp/data/final_manifests
  python prep_ksc2_libs.py join-gold --lib1 <库1过滤csv> --gold ... --outdir ...
  python prep_ksc2_libs.py q2-prep --q2 <q2 metadata csv> --outdir ...
"""
import argparse, csv, os, random, sys

csv.field_size_limit(sys.maxsize)
EXPECT_POOL = 627_824   # ksc2_train_pool_v2.csv 全量口径
EXPECT_Q2 = 628_002     # q2 metadata 全量口径


def read_csv(p):
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows, p, fieldnames):
    with open(p, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {p}")


def mode_merge(a):
    s0, s1 = read_csv(a.shard0), read_csv(a.shard1)
    rows = s0 + s1
    print(f"shard0={len(s0)} shard1={len(s1)} 合计={len(rows)}（期望 {EXPECT_POOL}）")
    if len(rows) < EXPECT_POOL * 0.99:
        sys.exit(f"FATAL: 合并行数损失 >1%（{len(rows)}/{EXPECT_POOL}），先查解码跳过日志")
    seen, uniq = set(), []
    for r in rows:
        if r["file_path"] in seen:
            sys.exit(f"FATAL: file_path 重复 {r['file_path']}")
        seen.add(r["file_path"])
        uniq.append(r)
    write_csv(uniq, f"{a.outdir}/e12_ksc2_pool_full.csv", list(rows[0].keys()))
    print(f"库1f（全量伪标）输入时长 {sum(float(r['duration']) for r in uniq)/3600:.1f}h")


def mode_join_gold(a):
    lib1 = read_csv(a.lib1)
    gold = {r["file_path"]: r for r in read_csv(a.gold)}
    missing = [r["file_path"] for r in lib1 if r["file_path"] not in gold]
    if missing:
        sys.exit(f"FATAL: {len(missing)} 条库1音频在金标 manifest 缺失（应 1:1），首例 {missing[0]}")
    lib4 = [{"file_path": r["file_path"], "sentence": gold[r["file_path"]]["sentence"],
             "duration": r["duration"], "source": gold[r["file_path"]].get("source", ""),
             "subdomain": gold[r["file_path"]].get("subdomain", "")} for r in lib1]
    write_csv(lib4, f"{a.outdir}/ksc2_lib4_gold.csv",
              ["file_path", "sentence", "duration", "source", "subdomain"])
    same_dur = sum(1 for r in lib4
                   if abs(float(r["duration"]) - float(gold[r["file_path"]]["duration"])) < 0.01)
    print(f"时长逐行一致 {same_dur}/{len(lib4)}；库4（=库1同音频金标）时长 "
          f"{sum(float(r['duration']) for r in lib4)/3600:.1f}h（库1 过滤后 "
          f"{sum(float(r['duration']) for r in lib1)/3600:.1f}h）")


def mode_q2_prep(a):
    rows = read_csv(a.q2)
    if len(rows) != EXPECT_Q2:
        sys.exit(f"FATAL: q2 行数 {len(rows)} ≠ {EXPECT_Q2}")
    out = [{"file_path": r["file_path"], "wrd": r["sentence"], "duration": r["duration"],
            "avg_logprob": r["avg_logprob"], "no_speech_prob": r["no_speech_prob"]} for r in rows]
    idx = list(range(len(out)))
    random.Random(0).shuffle(idx)
    miss = [out[i]["file_path"] for i in idx[:50] if not os.path.exists(out[i]["file_path"])]
    if miss:
        sys.exit(f"FATAL: q2 音频抽查 50 条缺失 {len(miss)}，首例 {miss[0]}")
    write_csv(out, f"{a.outdir}/yt_lib3p_input.csv",
              ["file_path", "wrd", "duration", "avg_logprob", "no_speech_prob"])
    print(f"q2 音频抽查 50/50 ✓；库3' 过滤输入时长 {sum(float(r['duration']) for r in out)/3600:.1f}h")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="mode", required=True)
m1 = sub.add_parser("merge")
m1.add_argument("--shard0", required=True)
m1.add_argument("--shard1", required=True)
m1.add_argument("--outdir", required=True)
m2 = sub.add_parser("join-gold")
m2.add_argument("--lib1", required=True)
m2.add_argument("--gold", required=True)
m2.add_argument("--outdir", required=True)
m3 = sub.add_parser("q2-prep")
m3.add_argument("--q2", required=True)
m3.add_argument("--outdir", required=True)
a = ap.parse_args()
{"merge": mode_merge, "join-gold": mode_join_gold, "q2-prep": mode_q2_prep}[a.mode](a)

