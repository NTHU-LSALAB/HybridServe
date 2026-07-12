# HybridServe

Implementation and evaluation for the thesis **HybridServe**, a study of
KV-cache transfer strategies for disaggregated prefill/decode LLM serving,
built on top of [vLLM](https://github.com/vllm-project/vllm) v0.8.3.

In disaggregated serving the prefill and decode stages run on separate GPUs (or
nodes). The KV cache produced during prefill must be moved to the decode worker.
*How* that transfer happens is the subject of this work; HybridServe compares
three strategies, each implemented on its own branch.

## Modes (branches)

| Branch        | Mode         | Idea |
|---------------|--------------|------|
| `pull_mode`   | Pull (baseline) | Stock vLLM 0.8.3 disaggregated prefill. The decode worker pulls the KV cache from the producer's buffer. Under an undersized buffer the prefill worker blocks until the consumer drains it. |
| `push_mode`   | Push         | The prefill worker pushes the KV cache to the decode worker's buffer as soon as it is ready, then exits early instead of waiting to be drained. |
| `hybridserve` | Hybrid (CPU-staged push) | Push-based transfer that stages the KV cache through a **CPU pinned-buffer pool** with non-blocking H2D/D2H copies, decoupling GPU buffer pressure from transfer latency. This is the thesis's main contribution and the default branch. |

```bash
git checkout hybridserve   # or push_mode / pull_mode
```

Each branch is a full vLLM fork with the mode's engine changes plus the
benchmark harness in [`ryan_bench/`](ryan_bench/). Pick a branch, build vLLM
from it, then run the benchmarks below.

> The core KV-transfer logic lives in
> `vllm/distributed/kv_transfer/kv_lookup_buffer/simple_buffer.py`.

## Environment setup

The reference environment is Python 3.10 + CUDA 12.4 + PyTorch 2.6.0, captured in
[`environment.yml`](environment.yml).

```bash
# 1. Create the conda env (installs torch, transformers, and all deps)
conda env create -f environment.yml
conda activate vllm_env

# 2. Build vLLM from this repo's checked-out branch.
#    VLLM_USE_PRECOMPILED reuses upstream compiled kernels, so only the Python
#    layer is (re)built -- fast, and enough for the changes in this fork.
VLLM_USE_PRECOMPILED=1 pip install -e .
```

Notes:
- `environment.yml` pins exact versions from the working `vllm_env`. The `vllm`
  line is intentionally commented out there — vLLM is installed from source in
  step 2, not from PyPI.
- PyTorch CUDA 12.4 wheels are pulled from the pytorch index (already wired into
  `environment.yml` via `--extra-index-url`).
- Two GPUs are needed to run a prefill + decode pair on one machine.

Verify:
```bash
python -c "import vllm; print(vllm.__version__)"   # -> 0.8.3+...
```

## Running the benchmarks

All benchmarks live in [`ryan_bench/`](ryan_bench/) — see
[`ryan_bench/README.md`](ryan_bench/README.md) for full details. Quick start:

Each branch ships **only the offline script its engine can run** (a push client
deadlocks on the pull engine, and vice-versa):

```bash
# On pull_mode:
python ryan_bench/offline/pull_offline_benchmark.py

# On push_mode / hybridserve:
python ryan_bench/offline/push_offline_benchmark.py

# Online (OpenAI-compatible serving; run each in its own shell)
./ryan_bench/online/prefill.sh
./ryan_bench/online/decode.sh
python ryan_bench/online/proxy_server.py
./ryan_bench/online/benchmark_serving.sh
```

Both offline scripts set `NCCL_P2P_DISABLE=1` internally (needed on GPUs without
NVLink, e.g. V100 PCIe); export `NCCL_P2P_DISABLE=0` to override.

To compare modes, check out a different branch, rebuild vLLM
(`VLLM_USE_PRECOMPILED=1 pip install -e .`), and run its matching benchmark.

## Repository layout

```
environment.yml                # reproducible conda environment
ryan_bench/                     # benchmark harness (see its README)
  offline/                      #   two-process prefill+decode benchmarks
  online/                       #   API-server + proxy serving benchmarks
vllm/                           # the vLLM engine (forked; mode-specific per branch)
```

This README documents the HybridServe fork; the original vLLM project README is
preserved as [`README_vllm.md`](README_vllm.md).
