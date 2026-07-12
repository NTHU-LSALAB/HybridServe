# Shared configuration for the online disaggregated-serving scripts.
# Source this from prefill.sh / decode.sh, or override any value from your shell:
#   MODEL=... KV_IP=10.0.0.1 ./prefill.sh
#
# All values have single-node (localhost) defaults so the scripts run on one
# box out of the box. For a multi-node setup, set KV_IP / *_HOST to the real
# addresses and NCCL_SOCKET_IFNAME to the NIC used for KV transfer.

# Model served by both prefill and decode nodes.
export MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"

# GPU to place each node on (single-node defaults; change per host).
export PREFILL_CUDA_DEVICE="${PREFILL_CUDA_DEVICE:-0}"
export DECODE_CUDA_DEVICE="${DECODE_CUDA_DEVICE:-1}"

# Ports for the two OpenAI-compatible API servers.
export PREFILL_PORT="${PREFILL_PORT:-8100}"
export DECODE_PORT="${DECODE_PORT:-8200}"

# KV-transfer rendezvous IP (the prefill node's address) and buffer size.
export KV_IP="${KV_IP:-127.0.0.1}"
export KV_BUFFER_SIZE="${KV_BUFFER_SIZE:-500}"

# NIC used by NCCL for the KV transfer. `ip -o link` lists interfaces.
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"

export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"
