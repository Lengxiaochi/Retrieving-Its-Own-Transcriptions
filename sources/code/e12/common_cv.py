# -*- coding: utf-8 -*-
"""E12：Common Voice 数据加载（norm 与第一篇口径一致）。
用法：from common_cv import load_cv; load_cv("tr", "test") → [(uid, wav, ref_normed)]"""
import csv
import os
import sys

CV_ROOT = "/root/autodl-tmp/data/CommonVoice/cv-corpus-26.0-2026-06-12"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e1"))
from common import norm  # noqa: E402


def load_cv(lang: str, split: str, limit: int = 0, hours: float = 0.0):
    """返回 [(uid, wav_path, ref_normed)]。split ∈ {test, dev, train, pool}。
    pool = validated+other 合并、去 dev/test（建库用全量池；E12 设计 §8）。"""
    base = f"{CV_ROOT}/{lang}"
    durs = {}
    with open(f"{base}/clip_durations.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            durs[r["clip"]] = float(r["duration[ms]"]) / 1000.0
    if split == "pool":
        # 审阅修复 P0-①（2026-09-24）：pool 减 dev/test（原注释写"去 dev/test"但实现未做）
        held, held_spk = set(), set()
        for sp in ("dev", "test"):
            with open(f"{base}/{sp}.tsv", encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                    held.add(r["path"]); held_spk.add(r["client_id"])
        out = []
        seen = set()
        for sp in ("validated", "other"):
            with open(f"{base}/{sp}.tsv", encoding="utf-8") as f:
                for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
                    if r["path"] in seen or r["path"] in held or r["client_id"] in held_spk:  # 去重；排除评测 clip 与评测说话人
                        continue
                    seen.add(r["path"])
                    out.append((r["path"], f"{base}/clips/{r['path']}", norm(r["sentence"], lang=lang)))
        out = [x for x in out if x[2]]
        if hours and hours > 0:
            picked, tot = [], 0.0
            for x in out:
                picked.append(x)
                tot += durs.get(x[0], 0)
                if tot >= hours * 3600:
                    break
            out = picked
        return out[:limit] if limit else out
    out = []
    seen_uids, excluded = set(), []
    with open(f"{base}/{split}.tsv", encoding="utf-8") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if None in r or r.get("sentence") is None or not r.get("path"):
                raise ValueError(f"Malformed TSV row in {lang}/{split}: {r}")
            if r["path"] in seen_uids:
                raise ValueError(f"Duplicate clip: {r['path']}")
            seen_uids.add(r["path"])
            # 路径状坏参考剔除（mn dev.tsv 实测 2 行 sentence 字段为文件路径，2026-09-12）
            sent = r["sentence"]
            if "common_voice" in sent or sent.startswith(("root ", "/", ".")):
                excluded.append((r["path"], "path_like_reference"))
                continue
            uid = r["path"]
            wav = f"{base}/clips/{r['path']}"
            ref = norm(sent, lang=lang)
            out.append((uid, wav, ref))
    excluded.extend((x[0], "empty_reference") for x in out if not x[2])
    out = [x for x in out if x[2]]
    if excluded:
        print(f"[reference exclusions] {lang}/{split}: {excluded}", flush=True)
    if lang == "tr" and split in ("dev", "test"):
        expected = {"dev": 11868, "test": 11869}[split]
        if len(out) != expected:
            raise ValueError(f"Turkish {split} coverage {len(out)} != {expected}")
    if hours and hours > 0:  # 按时长截断（保序）
        picked, tot = [], 0.0
        for x in out:
            picked.append(x)
            tot += durs.get(x[0], 0)
            if tot >= hours * 3600:
                break
        out = picked
    return out[:limit] if limit else out


def load_fleurs(lang: str, split: str = "test", limit: int = 0):
    """FLEURS 评测集加载（无表头 tsv：id, filename, raw, normed, chars, frames, gender）。
    音频在 /root/autodl-tmp/share/fleurs/<lang>/audio/<split>/<filename>。"""
    base = f"/root/autodl-tmp/share/fleurs/{lang}"
    out = []
    with open(f"{base}/trans/{split}.tsv", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            fid, ref = parts[1], parts[2]
            ref_n = norm(ref, lang=lang.split("_")[0])
            if not ref_n:
                continue
            out.append((fid, f"{base}/audio/{split}/{fid}", ref_n))
    return out[:limit] if limit else out
