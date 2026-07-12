# SPDX-License-Identifier: Apache-2.0
"""
PULL-based offline disaggregated-prefill benchmark.

Run this on the `pull_mode` branch (baseline vLLM). The prefill node stays alive
after producing the KV cache and the decode node PULLS it on demand -- so this
script deadlocks against a push engine. Use push_offline_benchmark.py on the
`push_mode` / `hybridserve` branches instead.

Launches two single-GPU vLLM instances in separate processes:
  * a prefill node   (kv_producer, GPU 0) that runs prompts with max_tokens=1
  * a decode  node   (kv_consumer, GPU 1) that pulls the KV cache and decodes

Consolidates the old gpu_buffer / insufficient_buffer / cpu_buffer benchmarks --
they only differed in the KV buffer device/size and the model. Select via flags:

  # GPU buffer, sufficient size (default)
  python pull_offline_benchmark.py

  # GPU buffer, deliberately too small -> prefill blocks (insufficient-buffer demo).
  # Add nvtx ranges in simple_buffer.py to observe the wait time.
  python pull_offline_benchmark.py --kv-buffer-size 160000

  # CPU buffer
  python pull_offline_benchmark.py \
      --buffer-device cpu \
      --model mistralai/Mistral-7B-Instruct-v0.2 \
      --max-model-len 200 --gpu-memory-utilization 0.95 --enforce-eager
"""

import argparse
import os
# These NCCL settings let the prefill<->decode KV transfer work on GPUs without
# NVLink (e.g. V100 PCIe). Set before vLLM/NCCL import. Override by exporting
# your own values (e.g. NCCL_P2P_DISABLE=0 on NVLink boxes).
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("NCCL_NET_GDR_LEVEL", "0")

import time
from multiprocessing import Process

import torch

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

PROMPTS = [
    "The cat sat on mat",
    "Five dogs ran past me",
    "She walked through the door",
    "He jumped over the fence",
    "They danced in the rain",
]


def build_kv_transfer_config(role: str, rank: int, args) -> KVTransferConfig:
    kwargs = dict(
        kv_connector="PyNcclConnector",
        kv_role=role,
        kv_rank=rank,
        kv_parallel_size=2,
        kv_buffer_device=args.buffer_device,
    )
    # Only pin the buffer size when explicitly requested; otherwise vLLM picks
    # a sufficiently large default.
    if args.kv_buffer_size is not None:
        kwargs["kv_buffer_size"] = args.kv_buffer_size
    return KVTransferConfig(**kwargs)


def build_llm(ktc: KVTransferConfig, args) -> LLM:
    return LLM(
        model=args.model,
        kv_transfer_config=ktc,
        max_model_len=args.max_model_len,
        dtype="half",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
    )


def run_prefill(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.prefill_gpu)

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    ktc = build_kv_transfer_config("kv_producer", 0, args)
    llm = build_llm(ktc, args)

    for i, prompt_text in enumerate(PROMPTS):
        torch.cuda.nvtx.range_push(f"prefill {i}")
        llm.generate([prompt_text], sampling_params)
        torch.cuda.nvtx.range_pop()

    print("Prefill node is finished with all prompts.")

    # Keep the prefill node alive until the decode node is done; otherwise the
    # process may exit before the KV cache transfer completes.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Script stopped by user.")


def run_decode(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.decode_gpu)

    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.95,
        min_tokens=args.output_len,
        max_tokens=args.output_len + 10,
    )
    ktc = build_kv_transfer_config("kv_consumer", 1, args)
    llm = build_llm(ktc, args)

    all_outputs = []
    for i, prompt_text in enumerate(PROMPTS):
        torch.cuda.nvtx.range_push(f"decode {i}")
        outputs = llm.generate([prompt_text], sampling_params)
        torch.cuda.nvtx.range_pop()
        all_outputs.extend(outputs)

    print("\n--- Decode Node: Final Outputs ---")
    for output in all_outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--buffer-device", choices=["cuda", "cpu"], default="cuda",
                   help="Device used to stage the KV buffer.")
    p.add_argument("--kv-buffer-size", type=int, default=None,
                   help="Pin the KV buffer size (bytes). Omit for vLLM default; "
                        "set small (e.g. 160000) to demo insufficient-buffer blocking.")
    p.add_argument("--max-model-len", type=int, default=2000)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--output-len", type=int, default=90,
                   help="Minimum decode tokens per prompt.")
    p.add_argument("--prefill-gpu", type=int, default=0)
    p.add_argument("--decode-gpu", type=int, default=1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prefill_process = Process(target=run_prefill, args=(args,))
    decode_process = Process(target=run_decode, args=(args,))

    prefill_process.start()
    decode_process.start()

    # Terminate the prefill node once decode has finished.
    decode_process.join()
    prefill_process.terminate()
