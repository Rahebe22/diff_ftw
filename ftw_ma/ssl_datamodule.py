import math
import mmap
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import kornia.augmentation as K
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Sampler, get_worker_info

from .dataset import load_image


def _distributed_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


class DiagnosticDataLoader(DataLoader):
    """Report progress while the first lazily loaded batch is prepared."""

    def __init__(
        self,
        *args,
        startup_diagnostics: bool = False,
        startup_log_interval_seconds: float = 30,
        split: str = "train",
        **kwargs,
    ) -> None:
        if startup_log_interval_seconds <= 0:
            raise ValueError("startup_log_interval_seconds must be positive")
        self.startup_diagnostics = startup_diagnostics
        self.startup_log_interval_seconds = startup_log_interval_seconds
        self.split = split
        self._reported_startup = False
        super().__init__(*args, **kwargs)

    def __iter__(self):
        if not self.startup_diagnostics or self._reported_startup:
            yield from super().__iter__()
            return

        rank = _distributed_rank()
        start = time.monotonic()
        print(
            f"[SSLDataModule] rank={rank} waiting for first {self.split} batch",
            flush=True,
        )
        stop_heartbeat = threading.Event()
        heartbeat = None
        if rank == 0:
            heartbeat = threading.Thread(
                target=self._log_waiting_heartbeat,
                args=(stop_heartbeat, start),
                daemon=True,
            )
            heartbeat.start()

        try:
            iterator = super().__iter__()
            first_batch = next(iterator)
        except StopIteration:
            self._reported_startup = True
            elapsed = time.monotonic() - start
            print(
                f"[SSLDataModule] rank={rank} {self.split} dataloader is empty "
                f"after {elapsed:.2f}s",
                flush=True,
            )
            return
        finally:
            stop_heartbeat.set()
            if heartbeat is not None:
                heartbeat.join()

        self._reported_startup = True
        elapsed = time.monotonic() - start
        print(
            f"[SSLDataModule] rank={rank} first {self.split} batch ready: "
            f"{elapsed:.2f}s",
            flush=True,
        )
        yield first_batch
        yield from iterator

    def _log_waiting_heartbeat(
        self,
        stop_heartbeat: threading.Event,
        start: float,
    ) -> None:
        while not stop_heartbeat.wait(self.startup_log_interval_seconds):
            elapsed = time.monotonic() - start
            print(
                f"[SSLDataModule] rank=0 still waiting for first {self.split} "
                f"batch: {elapsed:.0f}s elapsed",
                flush=True,
            )


class MMapSSLPaths:
    """Read cached relative image paths without expanding them into Python objects."""

    def __init__(self, cache_dir: str, split: str) -> None:
        self.cache_dir = Path(cache_dir)
        self.split = split
        self._open()

    def _open(self) -> None:
        paths_path = self.cache_dir / f"{self.split}.paths"
        offsets_path = self.cache_dir / f"{self.split}.offsets.u64"
        if not paths_path.exists() or not offsets_path.exists():
            raise FileNotFoundError(
                f"Missing SSL catalog cache for split '{self.split}' in "
                f"{self.cache_dir}. Run scripts/build_ssl_catalog_cache.py first."
            )
        self._offsets = np.memmap(offsets_path, dtype="<u8", mode="r")
        if self._offsets.size == 0:
            raise ValueError(f"SSL catalog cache offsets are empty: {offsets_path}")
        self._paths_file = paths_path.open("rb")
        self._paths = mmap.mmap(self._paths_file.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self) -> int:
        return self._offsets.size - 1

    def __getitem__(self, index: int) -> tuple[str, ...]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = int(self._offsets[index])
        end = int(self._offsets[index + 1])
        return tuple(self._paths[start:end].rstrip(b"\n").decode().split("\t"))

    def __getstate__(self) -> dict[str, Any]:
        return {"cache_dir": self.cache_dir, "split": self.split}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.cache_dir = state["cache_dir"]
        self.split = state["split"]
        self._open()

    def __del__(self) -> None:
        paths = getattr(self, "_paths", None)
        if paths is not None:
            paths.close()
        paths_file = getattr(self, "_paths_file", None)
        if paths_file is not None:
            paths_file.close()


class ConstantMemoryDistributedSampler(Sampler[int]):
    """Shuffle distributed samples with an affine permutation and O(1) memory."""

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        if num_replicas is None:
            num_replicas = dist.get_world_size() if dist.is_initialized() else 1
        if rank is None:
            rank = dist.get_rank() if dist.is_initialized() else 0
        if rank < 0 or rank >= num_replicas:
            raise ValueError(f"rank must be in [0, {num_replicas - 1}], got {rank}")
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0
        if drop_last:
            self.num_samples = len(dataset) // num_replicas
        else:
            self.num_samples = math.ceil(len(dataset) / num_replicas)
        self.total_size = self.num_samples * num_replicas

    def __iter__(self):
        size = len(self.dataset)
        if size == 0:
            return iter(())
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            offset = int(torch.randint(size, (1,), generator=generator).item())
            step = int(torch.randint(1, size + 1, (1,), generator=generator).item())
            while math.gcd(step, size) != 1:
                step = step % size + 1
        else:
            offset = 0
            step = 1
        return (
            (offset + (position % size) * step) % size
            for position in range(self.rank, self.total_size, self.num_replicas)
        )

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch


class FTWMapAfricaSSL(torch.utils.data.Dataset):
    """Image-only FTW dataset for diffusion self-supervised training."""

    valid_splits = ["train", "validate", "test"]

    def __init__(
        self,
        catalog: str,
        catalog_cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        split: str = "train",
        split_column: str = "split",
        img_path_cols: Optional[Union[str, List[str]]] = None,
        temporal_options: str = "windowB",
        num_samples: int = -1,
        normalization_strategy: str = "min_max",
        normalization_stat_procedure: str = "lab",
        global_stats: Optional[
            Union[Dict[str, Any], Tuple[Any, ...], List[Any]]
        ] = None,
        img_clip_val: float = 0,
        nodata: list = None,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
        loader_startup_diagnostics: bool = False,
    ) -> None:
        if split not in self.valid_splits:
            raise ValueError(f"split must be one of {self.valid_splits}, got {split}")

        self.data_dir = Path(data_dir) if data_dir is not None else Path(".")
        self.split_column = split_column
        self.img_path_cols = (
            [img_path_cols]
            if isinstance(img_path_cols, str)
            else img_path_cols
        )
        self.temporal_options = temporal_options
        self.normalization_strategy = normalization_strategy
        self.normalization_stat_procedure = normalization_stat_procedure
        self.global_stats = global_stats
        self.img_clip_val = img_clip_val
        self.nodata = nodata
        self.transforms = transforms
        self.loader_startup_diagnostics = loader_startup_diagnostics
        self._reported_first_image_read = False

        if catalog_cache_dir is not None:
            if self.img_path_cols is None:
                raise ValueError(
                    "catalog_cache_dir requires explicit img_path_cols"
                )
            self.filenames = MMapSSLPaths(catalog_cache_dir, split)
        else:
            self.filenames = self._read_catalog(catalog, split)
        self._length = len(self.filenames)
        if num_samples != -1:
            self._length = min(self._length, num_samples)

        print(f"Selecting {len(self)} {split} SSL samples")
        print(f"[SSLData] split_column={self.split_column}, img_path_cols={self.img_path_cols}")

    def _read_catalog(self, catalog: str, split: str) -> list[tuple[str, ...]]:
        path_cols = self.img_path_cols
        if path_cols is None:
            path_cols = ["window_a", "window_b"]
        required_cols = [self.split_column, *path_cols]
        catalog_df = pd.read_csv(catalog, usecols=required_cols)
        data = catalog_df.loc[catalog_df[self.split_column] == split, path_cols]
        filenames = []
        for row in data.itertuples(index=False, name=None):
            if self.img_path_cols is None:
                values = tuple(
                    None if pd.isna(value) else str(value) for value in row
                )
                paths = self._select_temporal_paths(values)
            else:
                paths = tuple(str(value) for value in row if not pd.isna(value))
            if paths:
                filenames.append(paths)
        return filenames

    def _select_temporal_paths(
        self, paths: tuple[Optional[str], ...]
    ) -> tuple[str, ...]:
        if self.temporal_options == "stacked":
            selected = paths
        elif self.temporal_options == "windowA":
            selected = paths[:1]
        elif self.temporal_options == "windowB":
            selected = paths[1:2]
        else:
            selected = ()
        return tuple(path for path in selected if path is not None)

    def __len__(self):
        return self._length

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        images = [
            self._load_image(filename)
            for filename in self.filenames[index]
        ]
        image = torch.from_numpy(np.concatenate(images, axis=0).astype("float32")).float()
        sample = {"image": image}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def _load_image(self, filename: str) -> np.ndarray:
        path = self.data_dir / filename
        report_read = (
            self.loader_startup_diagnostics and not self._reported_first_image_read
        )
        if report_read:
            self._reported_first_image_read = True
            worker = get_worker_info()
            worker_id = worker.id if worker is not None else "main"
            rank = _distributed_rank()
            start = time.monotonic()
            print(
                f"[SSLData] rank={rank} worker={worker_id} reading first image: "
                f"{path}",
                flush=True,
            )
        try:
            image = load_image(
                path,
                nodata_val_ls=self.nodata,
                apply_normalization=True,
                normal_strategy=self.normalization_strategy,
                stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                clip_val=self.img_clip_val,
            )
        except Exception:
            if report_read:
                elapsed = time.monotonic() - start
                print(
                    f"[SSLData] rank={rank} worker={worker_id} first image read "
                    f"failed after {elapsed:.2f}s: {path}",
                    flush=True,
                )
            raise
        if report_read:
            elapsed = time.monotonic() - start
            print(
                f"[SSLData] rank={rank} worker={worker_id} first image ready: "
                f"shape={image.shape}, {elapsed:.2f}s",
                flush=True,
            )
        return image


class FTWMapAfricaSSLDataModule(LightningDataModule):
    """FTW datamodule that returns image-only batches for diffusion SSL."""

    def __init__(
        self,
        catalog: str,
        catalog_cache_dir: Optional[str] = None,
        data_dir: Optional[str] = None,
        split_column: str = "split",
        img_path_cols: Optional[List[str]] = None,
        temporal_options: str = "windowB",
        num_samples: int = -1,
        img_clip_val: float = 0,
        nodata: Optional[List[float]] = None,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        drop_last_train: bool = True,
        use_constant_memory_sampler: bool = False,
        loader_startup_diagnostics: bool = False,
        loader_startup_log_interval_seconds: float = 30,
        global_stats: Optional[Dict[str, List[float]]] = None,
        normalization_strategy: str = "min_max",
        normalization_stat_procedure: str = "lab",
        crop_size: Optional[Union[int, Tuple[int, int]]] = None,
        use_augmentations: bool = True,
    ):
        super().__init__()
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        elif crop_size is None:
            crop_size = None
        else:
            crop_size = tuple(crop_size)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.drop_last_train = drop_last_train
        self.use_constant_memory_sampler = use_constant_memory_sampler
        self.loader_startup_diagnostics = loader_startup_diagnostics
        self.loader_startup_log_interval_seconds = (
            loader_startup_log_interval_seconds
        )
        self.global_stats = global_stats
        self.normalization_strategy = normalization_strategy
        self.normalization_stat_procedure = normalization_stat_procedure
        self.crop_size = crop_size
        self.use_augmentations = use_augmentations
        self.kwargs = {
            "catalog": catalog,
            "catalog_cache_dir": catalog_cache_dir,
            "data_dir": data_dir,
            "split_column": split_column,
            "img_path_cols": img_path_cols,
            "temporal_options": temporal_options,
            "num_samples": num_samples,
            "img_clip_val": img_clip_val,
            "nodata": nodata,
            "loader_startup_diagnostics": loader_startup_diagnostics,
        }

        augs = []
        if self.use_augmentations:
            augs.extend(
                [
                    K.RandomRotation(p=0.5, degrees=(90, 90)),
                    K.RandomHorizontalFlip(p=0.5),
                    K.RandomVerticalFlip(p=0.5),
                ]
            )
            if crop_size is not None:
                augs.append(
                    K.RandomResizedCrop(
                        size=crop_size,
                        scale=(0.3, 0.9),
                        ratio=(0.75, 1.33),
                        p=0.5,
                    )
                )
        self.train_aug = (
            K.AugmentationSequential(*augs, data_keys=["input"])
            if augs
            else None
        )
        print("[SSLDataModule] Initialized")
        print(f"[SSLDataModule] data_dir={self.kwargs.get('data_dir')}")
        print(f"[SSLDataModule] catalog={self.kwargs.get('catalog')}")
        print(f"[SSLDataModule] catalog_cache_dir={self.kwargs.get('catalog_cache_dir')}")
        print(f"[SSLDataModule] split_column={self.kwargs.get('split_column', 'split')}")
        print(f"[SSLDataModule] img_path_cols={self.kwargs.get('img_path_cols')}")
        print(f"[SSLDataModule] temporal_options={self.kwargs.get('temporal_options', 'windowB')}")
        print(f"[SSLDataModule] batch_size={self.batch_size}, num_workers={self.num_workers}")
        print(
            f"[SSLDataModule] pin_memory={self.pin_memory}, "
            f"prefetch_factor={self.prefetch_factor}, "
            f"drop_last_train={self.drop_last_train}"
        )
        print(
            "[SSLDataModule] use_constant_memory_sampler="
            f"{self.use_constant_memory_sampler}"
        )
        print(
            "[SSLDataModule] loader_startup_diagnostics="
            f"{self.loader_startup_diagnostics}, "
            "loader_startup_log_interval_seconds="
            f"{self.loader_startup_log_interval_seconds}"
        )
        print(
            f"[SSLDataModule] crop_size={self.crop_size}, "
            f"num_samples={self.kwargs.get('num_samples', -1)}"
        )
        print(f"[SSLDataModule] use_augmentations={self.use_augmentations}")

    def setup(self, stage: str):
        print(f"[SSLDataModule] setup(stage={stage})")
        if stage in ("fit", "train"):
            self.train_dataset = FTWMapAfricaSSL(
                split="train",
                normalization_strategy=self.normalization_strategy,
                normalization_stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                **self.kwargs,
            )
            print(f"[SSLDataModule] train samples={len(self.train_dataset)}")
        if stage in ("fit", "validate"):
            self.val_dataset = FTWMapAfricaSSL(
                split="validate",
                normalization_strategy=self.normalization_strategy,
                normalization_stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                **self.kwargs,
            )
            print(f"[SSLDataModule] validate samples={len(self.val_dataset)}")
        if stage in ("fit", "train", "validate"):
            train_count = len(getattr(self, "train_dataset", []))
            val_count = len(getattr(self, "val_dataset", []))
            print("[SSLDataModule] Dataset size summary")
            print(f"[SSLDataModule]   train: {train_count}")
            print(f"[SSLDataModule]   validate: {val_count}")
            print(f"[SSLDataModule]   batch_size: {self.batch_size}")
            if train_count:
                train_batches = (train_count + self.batch_size - 1) // self.batch_size
                print(f"[SSLDataModule]   train batches/epoch: {train_batches}")
            if val_count:
                val_batches = (val_count + self.batch_size - 1) // self.batch_size
                print(f"[SSLDataModule]   validate batches/epoch: {val_batches}")
        if stage == "test":
            self.test_dataset = FTWMapAfricaSSL(
                split="test",
                normalization_strategy=self.normalization_strategy,
                normalization_stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                **self.kwargs,
            )
            print(f"[SSLDataModule] test samples={len(self.test_dataset)}")

    def train_dataloader(self):
        print("[SSLDataModule] creating train dataloader")
        return self._dataloader(
            self.train_dataset,
            split="train",
            shuffle=True,
            drop_last=self.drop_last_train,
        )

    def val_dataloader(self):
        print("[SSLDataModule] creating validation dataloader")
        return self._dataloader(
            self.val_dataset,
            split="validation",
            shuffle=False,
        )

    def test_dataloader(self):
        return self._dataloader(
            self.test_dataset,
            split="test",
            shuffle=False,
        )

    def _dataloader(
        self,
        dataset,
        split: str,
        shuffle: bool,
        drop_last: bool = False,
    ) -> DataLoader:
        sampler = self._sampler(dataset, shuffle=shuffle)
        kwargs = self._loader_kwargs()
        loader = DataLoader
        if self.loader_startup_diagnostics:
            loader = DiagnosticDataLoader
            kwargs.update(
                {
                    "startup_diagnostics": True,
                    "startup_log_interval_seconds": (
                        self.loader_startup_log_interval_seconds
                    ),
                    "split": split,
                }
            )
        return loader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            drop_last=drop_last,
            **kwargs,
        )

    def _loader_kwargs(self) -> dict[str, Any]:
        kwargs = {
            "num_workers": self.num_workers,
            "persistent_workers": self.num_workers > 0,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def _sampler(self, dataset, shuffle: bool):
        if not self.use_constant_memory_sampler:
            return None
        return ConstantMemoryDistributedSampler(
            dataset,
            shuffle=shuffle,
            drop_last=shuffle and self.drop_last_train,
        )

    def on_after_batch_transfer(self, batch: dict[str, Tensor], dataloader_idx: int):
        if self.trainer and self.trainer.training and self.train_aug is not None:
            batch["image"] = self.train_aug(batch["image"])
        return batch
