# ryan_bench — HybridServe benchmarks

Benchmarks for disaggregated prefill/decode KV-cache transfer. There are two
families:

- **`offline/`** — self-contained scripts that spawn a prefill and a decode vLLM
  instance as two processes on one machine (GPU 0 = prefill, GPU 1 = decode).
  Best for quickly measuring transfer behavior without an HTTP stack.
- **`online/`** — an OpenAI-compatible serving setup (two API servers + a proxy)
  plus a load generator, for end-to-end TTFT / throughput measurements.

The KV-transfer *mode* is determined by which **branch** of this repo you build
vLLM from — see the top-level [README](../README.md). The engine's transfer
protocol differs per branch, so **you must run the offline script that matches
the branch**:

| Branch        | Engine protocol | Offline script to run          |
|---------------|-----------------|--------------------------------|
| `pull_mode`   | pull            | `offline/pull_offline_benchmark.py` |
| `push_mode`   | push            | `offline/push_offline_benchmark.py` |
| `hybridserve` | push (CPU-staged) | `offline/push_offline_benchmark.py` |

Both scripts are byte-identical on every branch (so the harness is uniform), but
a **push** client deadlocks against a **pull** engine and vice-versa — pick the
row above. Both set `NCCL_P2P_DISABLE=1` internally so they work on GPUs without
NVLink (e.g. V100 PCIe); export `NCCL_P2P_DISABLE=0` to override on NVLink boxes.

## Offline benchmarks

### `offline/pull_offline_benchmark.py`  (run on `pull_mode`)
Pull-based disaggregated prefill: the prefill node stays alive and the decode
node pulls the KV cache on demand. Consolidates the old `gpu_buffer_benchmark`,
`insufficient_buffer_benchmark`, and `cpu_buffer_benchmark` scripts (they only
differed in KV buffer device/size and model) into CLI flags:

```bash
# GPU buffer, sufficient size (default)
python offline/pull_offline_benchmark.py

# GPU buffer, deliberately too small -> prefill blocks.
# (Add nvtx ranges in simple_buffer.py to observe the wait time.)
python offline/pull_offline_benchmark.py --kv-buffer-size 160000

# CPU buffer
python offline/pull_offline_benchmark.py \
    --buffer-device cpu \
    --model mistralai/Mistral-7B-Instruct-v0.2 \
    --max-model-len 200 --gpu-memory-utilization 0.95 --enforce-eager
```

Run `--help` for all flags (model, GPUs, output length, ...).

### `offline/push_offline_benchmark.py`  (run on `push_mode` / `hybridserve`)
Push-based transfer: the prefill worker pushes KV caches to the decode worker
immediately and exits early. Adds QPS rate-limiting, per-request latency
tracking, and event-based shutdown. Tune the module-level constants at the top
of the file (`NUM_PROMPTS`, `QPS`, `BUFFER_SIZE`, ...), then:

```bash
python offline/push_offline_benchmark.py
```

## Online benchmarks (`online/`)

Three components; run each in its own shell. All configuration lives in
[`online/env.sh`](online/env.sh) with single-node localhost defaults — override
any value inline (e.g. `KV_IP=10.0.0.1 NCCL_SOCKET_IFNAME=eth0 ./prefill.sh`).

```bash
# 1. Prefill node (kv_producer)
./online/prefill.sh

# 2. Decode node (kv_consumer)
./online/decode.sh

# 3. Proxy that fans requests out to prefill then streams decode (port 8000)
python online/proxy_server.py
```

Then drive load or send a single request:

```bash
./online/benchmark_serving.sh          # vLLM benchmark_serving.py against the proxy
QPS=5 NUM_PROMPTS=100 ./online/benchmark_serving.sh
./online/curl.sh                       # single sample completion
```

Results land in `online/results/` as JSON (a couple of example runs are checked
in). `online/sonnet_4x.txt` is a sample dataset for the fixed-prompt mode.

### Multi-node
Set `KV_IP` to the prefill node's address on **both** nodes, put prefill and
decode on separate hosts, and set `NCCL_SOCKET_IFNAME` to the NIC used for KV
transfer (`ip -o link` lists interfaces). Point the proxy at the two servers via
`PREFILL_URL` / `DECODE_URL`.
