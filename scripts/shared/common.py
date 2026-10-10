# -*- coding: utf-8 -*-
"""E1 teacher 体检公共模块：数据加载 / 归一化 / WER / 幻觉指标。
归一化逻辑复用第一篇 active/cross_lingual/sanity_zeroshot_tt.py 的 norm()。"""
import json, re, unicodedata, zlib

OLD_PREFIX = "/home/24_linhy/dataset/ksc2/ISSAI_KSC2/"
NEW_PREFIX = "/root/autodl-tmp/share/ksc2/ISSAI_KSC2/"
KSC2_JSON = "/root/autodl-tmp/share/ksc2/ISSAI_KSC2/test.json"

PUNCT = re.compile(r"[^\w\sЀ-ӿ]")
SPACE = re.compile(r"\s+")
# 审阅修复（2026-09-24 P0-②）：旧 PUNCT 会抹掉泰米尔文组合符号（Mn 类：元音符/virama），
# 摧毁文字并改变词数。新法按 Unicode 类别保留 字母(L*)/附加符号(M*)/数字(N*)，其余作空白。
# Language-aware casing follows NFKC; Turkish I/ı and İ/i remain separate.
_KEEP_CATS = ("L", "M", "N")


NORMALIZATION_VERSION = "unicode-lmn-v3-20260925"

def norm(t: str, lang=None) -> str:
    # Canonical composition precedes casing: İ and I+combining-dot are identical.
    t = unicodedata.normalize("NFKC", str(t or ""))
    if lang == "tr":
        t = t.replace("I", "ı").replace("İ", "i")
    else:
        t = t.replace("İ", "I")
    t = t.lower()
    out = []
    for ch in t:
        out.append(ch if ch.isspace() or unicodedata.category(ch).startswith(_KEEP_CATS) else " ")
    t = "".join(out)
    toks = [w for w in SPACE.sub(" ", t).split() if any(c.isalnum() or unicodedata.category(c).startswith("M") for c in w)]
    return " ".join(toks)


def load_test(limit: int = 0):
    """返回 [(utt_id, wav_path, ref_normed)]，wav 路径已重映射到本机。"""
    d = json.load(open(KSC2_JSON, encoding="utf-8"))
    out = []
    for uid, v in d.items():
        wav = v["wav"].replace(OLD_PREFIX, NEW_PREFIX)
        with open(v["transcript"].replace(OLD_PREFIX, NEW_PREFIX), encoding="utf-8") as f:
            ref = norm(f.read())
        out.append((uid, wav, ref))
    out.sort(key=lambda x: x[0])
    return out[:limit] if limit else out


DEV_JSON = "/root/autodl-tmp/share/ksc2/ISSAI_KSC2/dev.json"


def load_dev(limit: int = 0):
    """KSC2 dev（8,685 句）。返回格式同 load_test。"""
    d = json.load(open(DEV_JSON, encoding="utf-8"))
    out = []
    for uid, v in d.items():
        wav = v["wav"].replace(OLD_PREFIX, NEW_PREFIX)
        with open(v["transcript"].replace(OLD_PREFIX, NEW_PREFIX), encoding="utf-8") as f:
            ref = norm(f.read())
        out.append((uid, wav, ref))
    out.sort(key=lambda x: x[0])
    return out[:limit] if limit else out


def compression_ratio(text: str) -> float:
    b = (text or "").encode("utf-8")
    return len(b) / max(1, len(zlib.compress(b)))


def non_cyrillic_word_ratio(text: str) -> float:
    """含拉丁字母的词占比——语言/文字混淆的代理指标（哈语正确输出不应含拉丁词）。"""
    words = (text or "").split()
    if not words:
        return 0.0
    return sum(1 for w in words if re.search(r"[A-Za-z]", w)) / len(words)


def score_and_save(data, hyps, out_path, model_name, raw_hyps=None, token_ids=None, metadata=None):
    """data: [(uid,wav,ref)]；hyps: 同长 list[str]（已 norm）。聚合 WER/CER＋幻觉指标并落盘。"""
    import jiwer
    if not data or len(data) != len(hyps) or len({x[0] for x in data}) != len(data):
        raise ValueError("Empty data, duplicate UID, or hypothesis coverage mismatch")
    for extra in (raw_hyps, token_ids):
        if extra is not None and len(extra) != len(data):
            raise ValueError("Raw output coverage mismatch")
    agg = {"ref_words": 0, "wer_errs": 0, "ref_chars": 0, "cer_errs": 0}
    per = {}
    empties = 0
    substitutions = deletions = insertions = 0
    for (uid, wav, ref), hyp in zip(data, hyps):
        if not hyp.strip():
            empties += 1
        m = jiwer.process_words(ref, hyp)
        substitutions += m.substitutions
        deletions += m.deletions
        insertions += m.insertions
        agg["ref_words"] += m.hits + m.substitutions + m.deletions
        agg["wer_errs"] += m.substitutions + m.deletions + m.insertions
        mc = jiwer.process_characters(ref, hyp)
        agg["ref_chars"] += mc.hits + mc.substitutions + mc.deletions
        agg["cer_errs"] += mc.substitutions + mc.deletions + mc.insertions
        per[uid] = {
            "hyp": hyp,
            "cr": round(compression_ratio(hyp), 3),
            "lat": round(non_cyrillic_word_ratio(hyp), 3),
        }
    for i, (uid, _, ref) in enumerate(data):
        per[uid]["ref"] = ref
        if raw_hyps is not None:
            per[uid]["raw_hyp"] = raw_hyps[i]
        if token_ids is not None:
            per[uid]["token_ids"] = token_ids[i]
    n = len(data)
    summary = {
        "normalization_version": NORMALIZATION_VERSION,
        "substitutions": substitutions, "deletions": deletions, "insertions": insertions,
        "ref_words": agg["ref_words"], "wer_errors": agg["wer_errs"],
        "deletion_rate": 100 * deletions / max(1, agg["ref_words"]),
        "model": model_name,
        "n": n,
        "WER": round(100 * agg["wer_errs"] / max(1, agg["ref_words"]), 2),
        "CER": round(100 * agg["cer_errs"] / max(1, agg["ref_chars"]), 2),
        "empty_rate": round(100 * empties / max(1, n), 2),
        "mean_CR": round(sum(v["cr"] for v in per.values()) / max(1, n), 3),
        "high_CR_rate": round(100 * sum(1 for v in per.values() if v["cr"] >= 2.4) / max(1, n), 2),
        "mean_latin_word_ratio": round(sum(v["lat"] for v in per.values()) / max(1, n), 4),
    }
    from provenance import atomic_json
    atomic_json(out_path, {"summary": summary, "per_utt": per, "metadata": metadata or {}})
    return summary

