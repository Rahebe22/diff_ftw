#!/usr/bin/env python3
"""Build compact path caches for large diffusion SSL catalogs."""

import argparse
import csv
import json
import struct
from pathlib import Path


class SplitCacheWriter:
    def __init__(self, output_dir: Path, split: str) -> None:
        self.paths_path = output_dir / f"{split}.paths"
        self.offsets_path = output_dir / f"{split}.offsets.u64"
        self.paths_tmp = output_dir / f"{split}.paths.tmp"
        self.offsets_tmp = output_dir / f"{split}.offsets.u64.tmp"
        self.paths = self.paths_tmp.open("wb")
        self.offsets = self.offsets_tmp.open("wb")
        self.offset = 0
        self.count = 0

    def write(self, paths: list[str]) -> None:
        payload = ("\t".join(paths) + "\n").encode()
        self.offsets.write(struct.pack("<Q", self.offset))
        self.paths.write(payload)
        self.offset += len(payload)
        self.count += 1

    def close(self, commit: bool) -> None:
        self.offsets.write(struct.pack("<Q", self.offset))
        self.paths.close()
        self.offsets.close()
        if commit:
            self.paths_tmp.replace(self.paths_path)
            self.offsets_tmp.replace(self.offsets_path)
        else:
            self.paths_tmp.unlink(missing_ok=True)
            self.offsets_tmp.unlink(missing_ok=True)


def build_cache(
    catalog: Path,
    output_dir: Path,
    split_column: str,
    path_columns: list[str],
    splits: list[str],
) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = {split: SplitCacheWriter(output_dir, split) for split in splits}
    completed = False
    try:
        with catalog.open(newline="") as source:
            reader = csv.DictReader(source)
            required = {split_column, *path_columns}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"Catalog columns missing: {sorted(missing)}")
            for row_number, row in enumerate(reader, start=1):
                split = row[split_column]
                if split in writers:
                    paths = [row[column] for column in path_columns if row[column]]
                    if paths:
                        writers[split].write(paths)
                if row_number % 1_000_000 == 0:
                    counts = ", ".join(
                        f"{name}={writer.count:,}"
                        for name, writer in writers.items()
                    )
                    print(f"Processed {row_number:,} rows: {counts}", flush=True)
        completed = True
    finally:
        for writer in writers.values():
            writer.close(commit=completed)

    counts = {split: writer.count for split, writer in writers.items()}
    metadata = {
        "catalog": str(catalog),
        "split_column": split_column,
        "path_columns": path_columns,
        "counts": counts,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split-column", default="usage")
    parser.add_argument("--path-columns", nargs="+", default=["image"])
    parser.add_argument("--splits", nargs="+", default=["train", "validate"])
    args = parser.parse_args()
    counts = build_cache(
        args.catalog,
        args.output_dir,
        args.split_column,
        args.path_columns,
        args.splits,
    )
    print("SSL catalog cache ready: " + ", ".join(
        f"{split}={count:,}" for split, count in counts.items()
    ))


if __name__ == "__main__":
    main()
