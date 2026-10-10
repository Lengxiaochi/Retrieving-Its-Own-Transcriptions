# Retrieving Its Own Transcriptions: Reproducibility Code

Code and analysis scripts for **Retrieving Its Own Transcriptions: Self-Generated
Memory for Low-Resource ASR**.

Repository: <https://github.com/Lengxiaochi/Retrieving-Its-Own-Transcriptions>

## What is included

The package contains the scripts used for the paper's main pipeline and
statistics:

- pseudo-transcript generation and quality filtering;
- datastore construction from decoder states and token values;
- FAISS IVFPQ indexing and retrieval-augmented decoding;
- no-retrieval baselines and matched-condition comparisons;
- datastore controls, subset construction, frequency/error analyses;
- paired bootstrap and other CPU-side statistical analyses.

The implementation follows the paper's pipeline:

```text
unlabeled audio -> Whisper pseudo-transcripts -> filtering
-> key/value datastore -> FAISS index -> kNN interpolation and safeguards
-> evaluation and statistical analysis
```

## Start here

Read these files in order:

1. [Paper-to-code map](docs/论文模块与代码对应.md)
2. [Configuration index](docs/配置索引.md)
3. [Run instructions](docs/运行说明.md)

`pipeline.py` provides the common entry point:

```bash
python3 pipeline.py list
python3 pipeline.py verify
python3 pipeline.py demo
```

`demo` is a small CPU-only teaching example. It is not a new paper result.

## Paths and data

The original speech corpora, model weights, datastore files, predictions, and
server-specific results are **not** included. Obtain KSC2, Common Voice,
FLEURS, and Whisper large-v3 from their original sources and follow their
licenses.

Copy `paths.example.json` to a local mapping file, fill in the local paths, and
prepare an isolated runtime:

```bash
python3 pipeline.py prepare --mapping paths.example.json
```

The scripts retain the input/output conventions used in the experiments, but
the complete GPU pipeline requires the listed model, data, FAISS, and
faster-Whisper dependencies. See [environment and dependencies](docs/环境与依赖.md).

## Reproduction scope

The release is intended to make the experiment code and statistical procedures
inspectable and reusable. It does not bundle large data, model weights,
predictions, or a prebuilt index. Full reproduction requires the corresponding
datasets, model files, hardware, and experiment-specific manifests.

The source package passed the release-time Python syntax and source-hash checks.
Those checks do not claim that the full GPU pipeline was rerun on a new machine.

## License and citation

The original datasets, model weights, and third-party libraries remain under
their own licenses. Before public redistribution, add the repository license
that covers the authors' code and retain the notices required by dependencies.

If you use this code, please cite the paper:

```bibtex
% Add the final bibliographic record after publication.
```

中文说明：本仓库公开论文实验使用的代码和统计脚本，不包含原始语音数据、模型权重、预测结果或服务器专用路径。运行前请按照 `paths.example.json` 配置本地路径，并阅读 `docs/运行说明.md`。完整 GPU 复现仍需要相应数据、模型、硬件和实验清单。
