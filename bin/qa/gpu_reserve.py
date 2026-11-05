#!/usr/bin/env python3
"""Keep a configurable amount of free memory occupied on selected CUDA GPUs."""
import argparse
import time
from typing import Dict, List
import torch

CHUNK_BYTES = 256 * 1024 * 1024

def parse_gpu_indices(value: str) -> List[int]:
    try:
        indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--gpus must be comma-separated integers") from exc
    if not indices:
        raise argparse.ArgumentTypeError("--gpus must contain at least one GPU index")
    if any(index < 0 for index in indices):
        raise argparse.ArgumentTypeError("GPU indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise argparse.ArgumentTypeError("GPU indices must not be repeated")
    return indices

def reserve(device_index: int, fraction: float) -> List[torch.Tensor]:
    device = torch.device(f"cuda:{device_index}")
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    except RuntimeError as exc:
        raise RuntimeError(f"cannot query GPU {device_index}: {exc}") from exc
    target = int(free_bytes * fraction)
    chunks: List[torch.Tensor] = []
    remaining = target
    try:
        while remaining:
            size = min(CHUNK_BYTES, remaining)
            chunks.append(torch.empty(size, dtype=torch.uint8, device=device))
            remaining -= size
    except RuntimeError as exc:
        del chunks
        if "out of memory" in str(exc).lower():
            raise RuntimeError(f"GPU {device_index} ran out of memory while reserving {target / 1024**3:.2f} GiB (free was {free_bytes / 1024**3:.2f} GiB)") from exc
        raise
    reserved = sum(chunk.numel() for chunk in chunks)
    print(f"GPU {device_index}: reserved {reserved / 1024**3:.2f} GiB of {total_bytes / 1024**3:.2f} GiB total", flush=True)
    return chunks

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=parse_gpu_indices, default=[0, 1], help="comma-separated GPU indices")
    parser.add_argument("--fraction", type=float, default=0.85, help="fraction of currently free memory to occupy")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; check the CUDA-enabled PyTorch environment and NVIDIA driver")
    count = torch.cuda.device_count()
    invalid = [index for index in args.gpus if index >= count]
    if invalid:
        raise SystemExit(f"GPU index {invalid[0]} is invalid; visible GPU count is {count}")
    if not 0 < args.fraction < 1:
        raise SystemExit("--fraction must be between 0 and 1")
    allocations: Dict[int, List[torch.Tensor]] = {}
    try:
        for index in args.gpus:
            allocations[index] = reserve(index, args.fraction)
        print("Reservation active; press Ctrl-C to release it.", flush=True)
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("\nReleasing reservations.", flush=True)
    finally:
        allocations.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
