# -*- coding: utf-8 -*-
"""失效模式三数口径（外部评估 P1-2 + dev-1k 禁令项④收口，2026-09-15）：
对每个失效模式子集报三数——①发生率（%句）②错误 token 占比（子集错误数/总错误数）
③WER 绝对贡献（点，剔除该子集后聚合 WER 的下降量）。
全量 KSC2 test（n=9,351），基线=e3full_test_lam00（λ0 裸）vs 终选=e3v21_test_lam05_retry（λ0.5 全栈）。
子集定义（对齐 E3 记录、以 base/final 两配置重述）：
  loop    = CR(base)≥2.4 或 CR(final)≥2.4（复读/跑飞灾难句）
  regress = 基线尚可(<40) 且 final 恶化 ≥20 点（真回退）
  win     = final 相对 base 改善 ≥30 点（大胜）
  digit   = 参考含数词且任一配置输出含数码（数字口径错位句）
用法：python failure_three_numbers.py [--split test]
通用化（2026-09-22，test 批次链用）：--base/--final 可覆盖两配置的 per_utt json
（如矩阵 S 点 kkm_base_test_lam0.0.json 与最优库 kkm_<tag>_test_lam<λ*>.json）；
缺省仍为 E3 原始两文件（裸 λ0 vs λ0.5 全栈），行为与 2026-09-15 版逐字一致。
"""
import argparse, json, os, sys, zlib
from collections import Counter

import jiwer

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "shared"))
from common import load_dev, load_test  # noqa: E402

R = os.path.join(os.path.dirname(os.path.dirname(CODE_DIR)), "results")
FILES = {"base": f"{R}/e3full_test_lam00.json", "final": f"{R}/e3v21_test_lam05_retry.json"}

ap = argparse.ArgumentParser()
ap.add_argument("--split", choices=["dev", "test"], default="test")
ap.add_argument("--cv-lang", default=None, help="CV 系语言码（tr/mn/...）——评测参考改用 CV 对应 split（2026-09-22 加）")
ap.add_argument("--base", default=None, help="覆盖 base 侧 per_utt json（缺省=E3 裸 λ0）")
ap.add_argument("--final", default=None, help="覆盖 final 侧 per_utt json（缺省=E3 λ0.5 全栈）")
args = ap.parse_args()
if args.base:
    FILES["base"] = args.base
if args.final:
    FILES["final"] = args.final


def cr(t):
    b = t.encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


def uwer(ref, hyp):
    if not ref.strip():
        return None
    m = jiwer.process_words(ref, hyp)
    tot = m.hits + m.substitutions + m.deletions
    if tot == 0:
        return None
    return 100 * (m.substitutions + m.deletions + m.insertions) / tot, \
        m.substitutions + m.deletions + m.insertions, tot


def has_numword(t):
    return any(w.replace("млн", "").replace("млрд", "").isalpha() and len(w) >= 2 and
               any(ch in "әғқңөұүһії" for ch in w) and False for w in t.split())  # 占位不启用


def has_digit(t):
    return any(any(c.isdigit() for c in w) for w in t.split())


if args.cv_lang:
    sys.path.insert(0, os.path.join(os.path.dirname(CODE_DIR), "pseudo_labels"))
    from common_cv import load_cv
    data = load_cv(args.cv_lang, args.split)
else:
    data = (load_dev if args.split == "dev" else load_test)()
refs = {uid: ref for uid, _, ref in data}
hyps = {c: {u: v["hyp"] for u, v in json.load(open(f))["per_utt"].items()} for c, f in FILES.items()}
uids = [u for u, _, _ in data if u in hyps["base"] and u in hyps["final"]]

per = {}
for u in uids:
    ref = refs[u]
    row = {}
    for c in ("base", "final"):
        r = uwer(ref, hyps[c][u])
        row[c] = {"wer": r[0] if r else None, "err": r[1] if r else 0, "den": r[2] if r else 0,
                  "cr": cr(hyps[c][u])}
    per[u] = row

# 数词集合（digit 子集判定用：参考写数词而输出数码）。按语言分立（2026-09-22 用户拍板
# "分别实现、别搞混"）：kk=原有硬编码表；kz=CV 内码即哈语，同 kk 表；tr=num2words 现成
# 词表运行时生成；mn=num2words 不支持→手写表。未覆盖语言报错退出，不许回退他语表。
NUMWORDS_KK = set("""бір екі үш төрт бес алты жеті сегіз тоғыз он онбір онекі онүш
жүз мың екімың миллион миллиард жиырма отыз қырық елу алпыс жетпіс сексен тоқсан""".split())
NUMWORDS_MN = set("""нэг хоёр гурав гурван дөрөв дөрвөн тав таван зургаа зурган
долоо долоон найм найман ес есөн арав арван хорин гучин дөчин тавин жаран далан наян
есөн зуун мянга мянган сая тэг""".split())


def _numwords_tr():
    from num2words import num2words
    words = set()
    for n in list(range(0, 21)) + list(range(20, 91, 10)) + [100, 1000]:
        words.update(num2words(n, lang="tr").split())
    words.add("milyon")   # num2words(10**6)='birmilyon' 连写——不补则参考里 "iki milyon" 检不出
    return words


_LANG = args.cv_lang or "kk"
if _LANG in ("kk", "kz"):
    NUMWORDS = NUMWORDS_KK
elif _LANG == "tr":
    NUMWORDS = _numwords_tr()
elif _LANG == "mn":
    NUMWORDS = NUMWORDS_MN
else:
    sys.exit(f"[abort] 数词表未覆盖 cv-lang={_LANG}——按防混纪律不回退他语表，补表后再跑")


def ref_has_numword(t):
    return any(w in NUMWORDS for w in t.lower().split())


subsets = {"loop": [], "regress": [], "win": [], "digit": []}
for u, p in per.items():
    b, f = p["base"], p["final"]
    if b["wer"] is None or f["wer"] is None:
        continue
    if max(b["cr"], f["cr"]) >= 2.4:
        subsets["loop"].append(u)
    elif b["wer"] < 40 and f["wer"] >= b["wer"] + 20:
        subsets["regress"].append(u)
    elif f["wer"] + 30 <= b["wer"]:
        subsets["win"].append(u)
    if ref_has_numword(refs[u]) and (has_digit(hyps["base"][u]) or has_digit(hyps["final"][u])):
        subsets["digit"].append(u)

n_all = sum(1 for p in per.values() if p["base"]["wer"] is not None and p["final"]["wer"] is not None)


def agg(cfg, drop=frozenset()):
    e = sum(p[cfg]["err"] for u, p in per.items() if p["base"]["wer"] is not None and u not in drop)
    d = sum(p[cfg]["den"] for u, p in per.items() if p["base"]["wer"] is not None and u not in drop)
    return 100 * e / d if d else 0.0


print(f"全量 {args.split}：n={n_all}")
print(f"{'子集':<8} {'句数':>5} {'发生率':>7} | {'base均WER':>9} {'final均WER':>10} | "
      f"{'错误占比(base→final)':>20} | {'WER贡献点(base→final)':>20}")
for name, us in subsets.items():
    if not us:
        print(f"{name:<8} {0:>5}")
        continue
    eb = sum(per[u]["base"]["err"] for u in us)
    ef = sum(per[u]["final"]["err"] for u in us)
    tb = sum(p["base"]["err"] for p in per.values() if p["base"]["wer"] is not None)
    tf = sum(p["final"]["err"] for p in per.values() if p["base"]["wer"] is not None)
    cb = agg("base") - agg("base", set(us))
    cf = agg("final") - agg("final", set(us))
    ab = sum(per[u]["base"]["wer"] for u in us) / len(us)
    af = sum(per[u]["final"]["wer"] for u in us) / len(us)
    print(f"{name:<8} {len(us):>5} {100*len(us)/n_all:>6.1f}% | {ab:>8.0f}->{af:<8.0f} | "
          f"{100*eb/tb:>9.1f}%->{100*ef/tf:<9.1f}% | {cb:>9.2f}->{cf:<9.2f}")
print(f"\n聚合 WER：base={agg('base'):.2f}  final={agg('final'):.2f}  "
      f"（总错误 token base={tb} final={tf}）")

