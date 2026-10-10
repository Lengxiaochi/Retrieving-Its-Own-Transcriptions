# Retrieving Its Own Transcriptions

Reproducibility scripts for **Retrieving Its Own Transcriptions: Self-Generated Memory for Low-Resource ASR**.

本仓库只发布论文实验所需的代码和统计脚本，不包含语音数据、模型权重、预测文件、datastore、FAISS 索引或服务器内部结果。

## Repository layout

```text
scripts/
  shared/          data loading, normalization, provenance helpers
  pseudo_labels/   pseudo-label generation, filtering, baselines, subsets
  retrieval/       datastore construction, FAISS indexing, kNN decoding
  analysis/        frequency, digit, failure, and auxiliary analyses
  statistics/      paired bootstrap and CPU-side statistics
  demo_metrics.py  small CPU-only demonstration
```

## Main pipeline

```text
unlabeled audio -> Whisper pseudo-transcripts -> filtering
-> token-level key/value datastore -> FAISS IVFPQ index
-> kNN interpolation and decoding safeguards -> scoring and analysis
```

Representative entry points are:

- `scripts/pseudo_labels/decode_ksc2_pool.py` and `decode_cv_strict.py`
- `scripts/pseudo_labels/filter_manifest.py`
- `scripts/retrieval/build_datastore.py`
- `scripts/retrieval/build_index.py`
- `scripts/retrieval/decode_knn.py`
- `scripts/pseudo_labels/std_baseline.py`
- `scripts/statistics/paired_stats.py`

The analysis scripts reproduce the paper's frequency, digit, error-case, and
CPU-side statistical summaries. They expect the input manifests and prediction
files described in the paper; those large files are intentionally not bundled.

## Environment and data

The experiments use Whisper large-v3, faster-Whisper/CTranslate2, PyTorch,
FAISS, and the corresponding dataset loaders. The exact versions, hardware,
dataset revisions, and experiment-specific manifests remain necessary for full
reproduction. Obtain KSC2, Common Voice, FLEURS, and model weights from their
original sources and follow their licenses.

```bash
python3 scripts/demo_metrics.py
python3 scripts/repro_check.py
```

`repro_check.py` checks Python syntax and reports the scripts included in the
release. It does not claim that a complete GPU reproduction has been run.

## Reproducibility boundary

The release makes the datastore construction, FAISS indexing, decoding,
baseline, scoring, and statistical procedures inspectable. Full reproduction
still requires the original datasets, model files, local manifests, suitable
hardware, and the command-line arguments described in the paper.

## Citation and license

Please cite the paper after its final bibliographic record is available.

The datasets, model weights, and third-party libraries retain their original
licenses. The repository license for the authors' scripts will be added as a
separate `LICENSE` file; until then, the code remains under the authors'
copyright. Do not redistribute the datasets or model weights through this
repository.

中文说明：本仓库公开论文实验代码和统计脚本，不包含原始语音数据、模型权重、预测文件、datastore 或 FAISS 索引。完整 GPU 复现需要按照论文说明准备数据、模型、依赖、硬件和实验参数。
