#!/usr/bin/env python3
"""Standalone LightningCLI launcher for single-node and DDP training."""

import os

from lightning import LightningModule
from lightning.pytorch.cli import LightningCLI


def main() -> None:
    os.environ.update(
        {
            "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
            "AWS_NO_SIGN_REQUEST": "YES",
            "GDAL_MAX_RAW_BLOCK_CACHE_SIZE": "200000000",
            "GDAL_SWATH_SIZE": "200000000",
            "VSI_CURL_CACHE_SIZE": "200000000",
        }
    )
    LightningCLI(
        model_class=LightningModule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
