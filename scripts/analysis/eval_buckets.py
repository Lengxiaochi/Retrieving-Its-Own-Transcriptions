# -*- coding: utf-8 -*-
"""E3 分桶评测：按池词频把参考词分桶（OOV/低频/中频/高频），词级对齐后统计每桶 WER。
用法：python eval_buckets.py --hyp ../../results/e3/dev1k_lam03.json [--freq word_freq.json]"""
import argparse, json, os, sys
from collections import Counter

import jiwer

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "shared"))
from common import load_dev, load_test  # noqa: E402

DS_FREQ = "/root/autodl-tmp/e3_ds/pilot30h/word_freq.json"

ap = argparse.ArgumentParser()
ap.add_argument("--hyp", required=True, help="score_and_save 输出的 results json")
ap.add_argument("--split", choices=["dev", "test"], default="dev")
ap.add_argument("--freq", default=DS_FREQ)
ap.add_argument("--quantiles", type=int, default=0,
                help=">0：词频分位分档追加表（词频≥1 的参考词按池频排序等样本量切 N 档；OOV 独立成档）")
ap.add_argument("--cv-lang", default=None, help="CV 系语言码（tr/mn/...）——评测参考改用 CV 对应 split")
ap.add_argument("--fine-bins", type=int, default=0,
                help=">0：连续细分档追加表（词频≥1 按 log10(池频) 等宽切 N 档；OOV 独立成档）")
args = ap.parse_args()

freq = Counter(json.load(open(args.freq, encoding="utf-8")))
res = json.load(open(args.hyp, encoding="utf-8"))["per_utt"]
if args.cv_lang:
    sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "pseudo_labels"))
    from common_cv import load_cv
    data = load_cv(args.cv_lang, args.split)
else:
    data = (load_dev if args.split == "dev" else load_test)()

def bucket(w: str) -> str:
    f = freq.get(w, 0)
    if f == 0:
        return "oov"        # 池文本未见（专名/稀疏）
    if f <= 5:
        return "low1-5"
    if f <= 100:
        return "mid6-100"
    return "high>100"

ORDER = ["oov", "low1-5", "mid6-100", "high>100"]
stat = {b: [0, 0] for b in ORDER}   # [ref_words, errors]
words_f_err = []                    # [(池频, 错误0/1)]——分位/细分的原料（默认档不消费，行为不变）
insertions = 0
n_used = 0
for uid, _, ref in data:
    if uid not in res:
        continue
    hyp = res[uid]["hyp"]
    n_used += 1
    m = jiwer.process_words(ref, hyp) if ref or hyp else None
    if m is None:
        continue
    insertions += m.insertions
    for chunks in m.alignments:            # jiwer: list（句）× list（chunk）
        for c in chunks:
            words = ref.split()[c.ref_start_idx:c.ref_end_idx]
            for w in words:
                f = freq.get(w, 0)
                err = 0 if c.type == "equal" else 1
                stat[bucket(w)][0] += 1
                stat[bucket(w)][1] += err
                words_f_err.append((f, err))

print(f"file={os.path.basename(args.hyp)}  utts={n_used}  freq_source={os.path.basename(args.freq)}")
print(f"{'bucket':<10} {'ref_words':>10} {'errors':>8} {'WER%':>8}")
tot_r = tot_e = 0
for b in ORDER:
    r, e = stat[b]
    tot_r += r
    tot_e += e
    print(f"{b:<10} {r:>10} {e:>8} {100*e/max(1,r):>8.2f}")
print(f"{'TOTAL':<10} {tot_r:>10} {tot_e:>8} {100*tot_e/max(1,tot_r):>8.2f}   insertions={insertions}")
out = {b: {"ref": r, "err": e, "wer": round(100*e/max(1, r), 2)} for b, (r, e) in stat.items()}
out["TOTAL"] = {"ref": tot_r, "err": tot_e, "wer": round(100*tot_e/max(1, tot_r), 2),
                "insertions": insertions}

# ---- 追加档（--quantiles / --fine-bins；分位=等样本量（对并列频率在排序内切分），细分=log10 等宽）----
import numpy as np
arr = np.array(words_f_err, dtype=np.int64)
if args.quantiles > 0 or args.fine_bins > 0:
    oov_mask = arr[:, 0] == 0
    oov_r, oov_e = int(oov_mask.sum()), int(arr[oov_mask, 1].sum())
    inv = arr[~oov_mask]
    inv = inv[np.argsort(inv[:, 0], kind="stable")]
if args.quantiles > 0:
    print(f"\n-- 分位分档（{args.quantiles} 档等样本量，词频≥1；OOV 独立）--")
    print(f"{'quantile':<10} {'ref_words':>10} {'errors':>8} {'WER%':>8}   池频范围")
    o = {"ref": oov_r, "err": oov_e, "wer": round(100*oov_e/max(1, oov_r), 2)}
    print(f"{'oov':<10} {oov_r:>10} {oov_e:>8} {o['wer']:>8.2f}   freq=0")
    qs = np.array_split(np.arange(len(inv)), args.quantiles)
    qout = {"oov": o}
    for i, idx in enumerate(qs, 1):
        seg = inv[idx]
        r, e = len(seg), int(seg[:, 1].sum())
        lo, hi = int(seg[:, 0].min()), int(seg[:, 0].max())
        w = round(100*e/max(1, r), 2)
        qout[f"q{i}"] = {"ref": r, "err": e, "wer": w, "freq": [lo, hi]}
        print(f"{'q'+str(i):<10} {r:>10} {e:>8} {w:>8.2f}   {lo}–{hi}")
    out["quantile"] = qout
if args.fine_bins > 0:
    print(f"\n-- 连续细分档（{args.fine_bins} 档 log10(池频) 等宽，词频≥1；OOV 独立）--")
    print(f"{'bin':<14} {'ref_words':>10} {'errors':>8} {'WER%':>8}   池频范围")
    o = {"ref": oov_r, "err": oov_e, "wer": round(100*oov_e/max(1, oov_r), 2)}
    print(f"{'oov':<14} {oov_r:>10} {oov_e:>8} {o['wer']:>8.2f}   freq=0")
    lg = np.log10(inv[:, 0])
    edges = np.linspace(lg.min(), lg.max(), args.fine_bins + 1)
    bid = np.clip(np.digitize(lg, edges[1:-1]), 0, args.fine_bins - 1)
    fout = {"oov": o}
    for i in range(args.fine_bins):
        seg = inv[bid == i]
        if len(seg) == 0:
            continue
        r, e = len(seg), int(seg[:, 1].sum())
        lo, hi = int(seg[:, 0].min()), int(seg[:, 0].max())
        w = round(100*e/max(1, r), 2)
        fout[f"b{i+1}"] = {"ref": r, "err": e, "wer": w, "freq": [lo, hi]}
        print(f"{f'b{i+1}':<14} {r:>10} {e:>8} {w:>8.2f}   {lo}–{hi}")
    out["fine"] = fout
json.dump(out, open(args.hyp.replace(".json", "_buckets.json"), "w"), indent=1)

