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
    if os.environ.get("FTW_DDP_DEBUG") == "1":
        debug_defaults = {
            "NCCL_DEBUG": "INFO",
            "NCCL_DEBUG_SUBSYS": "INIT,COLL",
            "TORCH_DISTRIBUTED_DEBUG": "DETAIL",
            "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
            "TORCH_NCCL_DESYNC_DEBUG": "1",
            "TORCH_NCCL_DUMP_ON_TIMEOUT": "1",
            "TORCH_NCCL_TRACE_BUFFER_SIZE": "200000",
        }
        for name, value in debug_defaults.items():
            os.environ.setdefault(name, value)
        print("[DDP diagnostics] Detailed PyTorch/NCCL logging enabled")
    LightningCLI(
        model_class=LightningModule,
        seed_everything_default=0,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
    )


if __name__ == "__main__":
    main()
