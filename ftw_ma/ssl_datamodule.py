from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import kornia.augmentation as K
import numpy as np
import pandas as pd
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from .dataset import load_image


class FTWMapAfricaSSL(torch.utils.data.Dataset):
    """Image-only FTW dataset for diffusion self-supervised training."""

    valid_splits = ["train", "validate", "test"]

    def __init__(
        self,
        catalog: str,
        data_dir: str = None,
        split: str = "train",
        split_column: str = "split",
        img_path_cols: Optional[Union[str, List[str]]] = None,
        temporal_options: str = "windowB",
        num_samples: int = -1,
        normalization_strategy: str = "min_max",
        normalization_stat_procedure: str = "lab",
        global_stats: Optional[Union[Dict[str, Any], Tuple, List]] = None,
        img_clip_val: float = 0,
        nodata: list = None,
        transforms: Optional[Callable[[dict[str, Tensor]], dict[str, Tensor]]] = None,
    ) -> None:
        if split not in self.valid_splits:
            raise ValueError(f"split must be one of {self.valid_splits}, got {split}")

        catalog_df = pd.read_csv(catalog)
        if split_column not in catalog_df.columns:
            raise ValueError(
                f"Catalog split column '{split_column}' not found. "
                f"Available columns: {list(catalog_df.columns)}"
            )

        self.data = catalog_df.loc[catalog_df[split_column] == split].copy()
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

        self.filenames = []
        if self.img_path_cols is not None:
            missing_cols = [
                col for col in self.img_path_cols if col not in self.data.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Catalog image path columns missing: {missing_cols}. "
                    f"Available columns: {list(self.data.columns)}"
                )
            for _, row in self.data.iterrows():
                paths = [self._to_path(row.get(col)) for col in self.img_path_cols]
                paths = [path for path in paths if path is not None]
                if paths:
                    self.filenames.append(paths)
        else:
            for _, row in self.data.iterrows():
                window_a = self._to_path(row.get("window_a"))
                window_b = self._to_path(row.get("window_b"))
                paths = []
                if self.temporal_options in ("stacked", "windowA") and window_a is not None:
                    paths.append(window_a)
                if self.temporal_options in ("stacked", "windowB") and window_b is not None:
                    paths.append(window_b)
                if paths:
                    self.filenames.append(paths)

        if num_samples != -1:
            self.filenames = self.filenames[:num_samples]

        print(f"Selecting {len(self.filenames)} {split} SSL samples")
        print(f"[SSLData] split_column={self.split_column}, img_path_cols={self.img_path_cols}")

    def _to_path(self, value):
        if pd.isna(value):
            return None
        return self.data_dir / str(value)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        images = [
            load_image(
                filename,
                nodata_val_ls=self.nodata,
                apply_normalization=True,
                normal_strategy=self.normalization_strategy,
                stat_procedure=self.normalization_stat_procedure,
                global_stats=self.global_stats,
                clip_val=self.img_clip_val,
            )
            for filename in self.filenames[index]
        ]
        image = torch.from_numpy(np.concatenate(images, axis=0).astype("float32")).float()
        sample = {"image": image}
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


class FTWMapAfricaSSLDataModule(LightningDataModule):
    """FTW datamodule that returns image-only batches for diffusion SSL."""

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        global_stats: Optional[Union[Dict[str, Any], Tuple, List]] = None,
        normalization_strategy: str = "min_max",
        normalization_stat_procedure: str = "lab",
        crop_size: Optional[Union[int, Tuple[int, int]]] = None,
        use_augmentations: bool = True,
        **kwargs,
    ):
        super().__init__()
        if "split" in kwargs:
            raise ValueError("Cannot specify split in FTWMapAfricaSSLDataModule")

        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        elif crop_size is None:
            crop_size = None
        else:
            crop_size = tuple(crop_size)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.global_stats = global_stats
        self.normalization_strategy = normalization_strategy
        self.normalization_stat_procedure = normalization_stat_procedure
        self.crop_size = crop_size
        self.use_augmentations = use_augmentations
        self.kwargs = kwargs

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
        print(f"[SSLDataModule] split_column={self.kwargs.get('split_column', 'split')}")
        print(f"[SSLDataModule] img_path_cols={self.kwargs.get('img_path_cols')}")
        print(f"[SSLDataModule] temporal_options={self.kwargs.get('temporal_options', 'windowB')}")
        print(f"[SSLDataModule] batch_size={self.batch_size}, num_workers={self.num_workers}")
        print(f"[SSLDataModule] crop_size={self.crop_size}, num_samples={self.kwargs.get('num_samples', -1)}")
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
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        print("[SSLDataModule] creating validation dataloader")
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def on_after_batch_transfer(self, batch: dict[str, Tensor], dataloader_idx: int):
        if self.trainer and self.trainer.training and self.train_aug is not None:
            batch["image"] = self.train_aug(batch["image"])
        return batch
