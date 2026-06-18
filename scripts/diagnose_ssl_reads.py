#!/usr/bin/env python3
"""Read a targeted portion of the distributed SSL epoch without GPUs/DDP."""

from __future__ import annotations

import argparse
import itertools
import os
import time
from pathlib import Path

from ftw_ma.ssl_datamodule import (
    ConstantMemoryDistributedSampler,
    FTWMapAfricaSSL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--catalog-cache-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--split-column", default="usage")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--start-batch", type=int, default=145800)
    parser.add_argument("--num-batches", type=int, default=501)
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    parser.add_argument("--slow-seconds", type=float, default=1)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = args.rank
    if rank is None:
        rank = int(os.environ.get("LOCAL_RANK", "0"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / f"data_rank_{rank}.log"
    log_path.write_text("", encoding="utf-8")

    dataset = FTWMapAfricaSSL(
        catalog=args.catalog,
        catalog_cache_dir=args.catalog_cache_dir,
        data_dir=args.data_dir,
        split=args.split,
        split_column=args.split_column,
        img_path_cols=[args.image_column],
        normalization_strategy="min_max",
        normalization_stat_procedure="lab",
        nodata=[],
        skip_bad_images=True,
        image_read_timeout_seconds=args.timeout_seconds,
        max_image_read_retries=8,
    )
    sampler = ConstantMemoryDistributedSampler(
        dataset,
        num_replicas=args.world_size,
        rank=rank,
        shuffle=True,
        seed=0,
        drop_last=True,
    )
    sampler.set_epoch(args.epoch)
    start_sample = args.start_batch * args.batch_size
    sample_count = args.num_batches * args.batch_size
    indices = itertools.islice(iter(sampler), start_sample, start_sample + sample_count)
    start = time.monotonic()
    processed = 0

    for position, index in enumerate(indices):
        read_start = time.monotonic()
        filenames = dataset.filenames[index]
        dataset[index]
        read_seconds = time.monotonic() - read_start
        if read_seconds >= args.slow_seconds or position % args.log_every == 0:
            line = (
                f"rank={rank} position={position} index={index} "
                f"read_seconds={read_seconds:.3f} files={filenames}\n"
            )
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(line)
            print(f"[Data diagnostic] {line.strip()}", flush=True)
        processed += 1

    elapsed = time.monotonic() - start
    line = (
        f"PASS rank={rank} samples={processed} "
        f"start_batch={args.start_batch} elapsed={elapsed:.2f}s\n"
    )
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(line)
    print(f"[Data diagnostic] {line.strip()}", flush=True)


if __name__ == "__main__":
    main()
