# -*- coding: utf-8 -*-
"""E3 极端样本分析：把 kNN 解码的灾难句 / 回退句 / 大胜句挑出来定量+定性刻画。
用法：python analyze_extremes.py [--split test]"""
import argparse, json, os, sys
from collections import Counter

import jiwer

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "shared"))
from common import load_dev, load_test, compression_ratio, non_cyrillic_word_ratio  # noqa: E402

R = "/root/autodl-tmp/dev/第二篇_跨领域低资源方法迁移/results"
ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--out", default=None)
args = ap.parse_args()

FILES = {  # 配置名 → results json
    "base": f"{R}/e3full_test_lam00.json",
    "v1": f"{R}/e3full_test_lam04_T100.json",
    "v2": f"{R}/e3v2_test_k16_cra18.json",
    "v21": f"{R}/e3v21_test_lam05_retry.json",
}
FILES = {k: v for k, v in FILES.items() if os.path.exists(v)}
print("configs:", list(FILES))
data = (load_dev if args.split == "dev" else load_test)()
refs = {uid: ref for uid, _, ref in data}
hyps = {cfg: {u: v["hyp"] for u, v in json.load(open(f))["per_utt"].items()}
        for cfg, f in FILES.items()}
uids = [u for u, _, _ in data if u in hyps["base"] and u in hyps["v1"]]


def uwer(ref, hyp):
    if not ref.strip():
        return None
    m = jiwer.process_words(ref, hyp)
    tot = m.hits + m.substitutions + m.deletions
    return None if tot == 0 else 100 * (m.substitutions + m.deletions + m.insertions) / tot


per = {}
for u in uids:
    ref = refs[u]
    e = {cfg: uwer(ref, hyps[cfg][u]) for cfg, _ in FILES.items()}
    per[u] = {
        "wer": e, "ref_len": len(ref), "ref_words": len(ref.split()) if ref else 0,
        "cr": {c: round(compression_ratio(hyps[c][u]), 2) for c in hyps},
        "lat": {c: round(non_cyrillic_word_ratio(hyps[c][u]), 3) for c in hyps},
    }

# ---------- 分三类 ----------
cat = {"loop": [], "regress": [], "win": []}
for u in uids:
    p = per[u]
    if p["wer"]["v1"] is None or p["wer"]["base"] is None:
        continue
    if max(p["cr"]["v1"], p["cr"]["v2"]) >= 2.4 or p["cr"].get("v21", 0) >= 2.4:
        cat["loop"].append(u)                       # 任一 kNN 配置复读
    elif (p["wer"]["v1"] is not None and p["wer"]["base"] + 20 <= p["wer"]["v1"]
          and p["wer"]["base"] < 40):
        cat["regress"].append(u)                    # 基线尚可(<40) 但 kNN 恶化 ≥20 点
    elif (p["wer"]["v1"] or 0) + 30 <= p["wer"]["base"]:
        cat["win"].append(u)                        # kNN 大胜 ≥30 点

def bucket_ref(u):
    words = refs[u].split()
    if not words:
        return "empty"
    lat = sum(1 for w in words if any(c.isalnum() and not w.isalpha() for c in w))
    return f"words={len(words)}"

print(f"\n=== 分类统计（{len(uids)} 句）===")
for name, us in cat.items():
    if not us:
        print(f"{name}: 0")
        continue
    n = len(us)
    w_base = sum(per[u]["wer"]["base"] for u in us) / n
    w_v1 = sum(per[u]["wer"]["v1"] for u in us) / n
    w_v21 = sum(per[u]["wer"].get("v21") or 0 for u in us) / n
    lat_drift = sum(1 for u in us if per[u]["lat"]["base"] > 0) / n
    long_ref = sum(1 for u in us if per[u]["ref_words"] > 25) / n
    print(f"{name}: {n} 句 ({100*n/len(uids):.1f}%) | 平均WER base={w_base:.0f} v1={w_v1:.0f} "
          f"v21={w_v21:.0f} | 基线拉丁词漂移率={lat_drift:.0%} | 长句(>25词)占比={long_ref:.0%}")

# ---------- 极端样本对聚合的贡献度 ----------
tot_ref = sum(len(refs[u].split()) for u in uids)
for name, us in cat.items():
    errs = 0
    for u in us:
        m = jiwer.process_words(refs[u], hyps["v1"][u])
        errs += m.substitutions + m.deletions + m.insertions
    print(f"{name}: v1 聚合 WER 错误贡献 = {errs} ({100*errs/max(1,sum(jiwer.process_words(refs[u], hyps['v1'][u]).substitutions + jiwer.process_words(refs[u], hyps['v1'][u]).deletions + jiwer.process_words(refs[u], hyps['v1'][u]).insertions for u in uids)):.1f}%)")

# ---------- 数字口径错位（ref 词数 vs hyp 数码）----------
import re
NUM_WORDS = {"бір", "екі", "үш", "төрт", "бес", "алты", "жеті", "сегіз", "тоғыз", "он",
             "жүз", "мың", "миллион", "миллиард"}
def has_numword(t):
    return any(w in NUM_WORDS for w in t.split())
def has_digit(t):
    return bool(re.search(r"\d", t))
mismatch = [u for u in uids
            if (has_numword(refs[u]) and has_digit(hyps["v1"][u]))
            or (has_digit(refs[u]) and has_numword(hyps["v1"][u]))]
err_mm = err_all = 0
for u in uids:
    m = jiwer.process_words(refs[u], hyps["v1"][u])
    e = m.substitutions + m.deletions + m.insertions
    err_all += e
    if u in set(mismatch):
        err_mm += e
print(f"\n=== 数字口径错位 ===")
print(f"含数字错位的句子: {len(mismatch)} ({100*len(mismatch)/len(uids):.1f}%)，"
      f"占 v1 聚合错误的 {100*err_mm/max(1,err_all):.1f}%")

# ---------- 样例 ----------
print("\n=== 样例（每类前 4）===")
for name, us in cat.items():
    for u in us[:4]:
        print(f"\n--- [{name}] {u} (ref_words={per[u]['ref_words']}, CR base/v1/v2/v21 = "
              f"{per[u]['cr']['base']}/{per[u]['cr']['v1']}/{per[u]['cr']['v2']}/{per[u]['cr'].get('v21','-')})")
        print(f"  REF : {refs[u][:150]}")
        for c in ["base", "v1", "v2", "v21"]:
            if c in hyps and u in hyps[c]:
                print(f"  {c:>4}: {hyps[c][u][:150]}")

out = args.out or f"{R}/e3_extremes_{args.split}.json"
json.dump({u: per[u] for u in uids}, open(out, "w", encoding="utf-8"), ensure_ascii=False)
print(f"\nsaved: {out}")

