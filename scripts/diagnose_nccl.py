#!/usr/bin/env python3
"""Exercise NCCL collectives without model or data loading."""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--large-elements", type=int, default=1_000_000)
    parser.add_argument("--large-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=5))
    rank = dist.get_rank()
    device = torch.device("cuda", local_rank)
    scalar = torch.ones(1, device=device)
    large = torch.ones(args.large_elements, device=device)
    start = time.monotonic()

    for iteration in range(args.iterations):
        dist.all_reduce(scalar)
        scalar.fill_(1)
        if args.large_every > 0 and iteration % args.large_every == 0:
            dist.all_reduce(large)
            large.fill_(1)
        if iteration % args.log_every == 0:
            torch.cuda.synchronize(device)
            elapsed = time.monotonic() - start
            print(
                f"[NCCL diagnostic] rank={rank} iteration={iteration} "
                f"elapsed={elapsed:.2f}s",
                flush=True,
            )

    dist.barrier()
    elapsed = time.monotonic() - start
    print(
        f"[NCCL diagnostic] PASS rank={rank} iterations={args.iterations} "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
