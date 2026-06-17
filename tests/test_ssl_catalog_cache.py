import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scripts.build_ssl_catalog_cache import build_cache
from ftw_ma.ssl_datamodule import (
    ConstantMemoryDistributedSampler,
    DiagnosticDataLoader,
    FTWMapAfricaSSL,
    FTWMapAfricaSSLDataModule,
    MMapSSLPaths,
)


def test_mmap_ssl_catalog_cache_reads_paths_by_split(tmp_path):
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "usage,image\n"
        "train,2017/train_a.tif\n"
        "validate,2017/validate_a.tif\n"
        "train,2017/train_b.tif\n"
    )
    cache_dir = tmp_path / "cache"

    counts = build_cache(
        catalog,
        cache_dir,
        split_column="usage",
        path_columns=["image"],
        splits=["train", "validate"],
    )

    assert counts == {"train": 2, "validate": 1}
    assert MMapSSLPaths(cache_dir, "train")[1] == ("2017/train_b.tif",)
    assert MMapSSLPaths(cache_dir, "validate")[0] == ("2017/validate_a.tif",)


def test_constant_memory_sampler_partitions_a_shuffled_epoch():
    dataset = range(10)
    rank_zero = ConstantMemoryDistributedSampler(
        dataset, num_replicas=2, rank=0, shuffle=True, seed=3
    )
    rank_one = ConstantMemoryDistributedSampler(
        dataset, num_replicas=2, rank=1, shuffle=True, seed=3
    )

    indices = list(rank_zero) + list(rank_one)

    assert sorted(indices) == list(dataset)


def test_loader_startup_diagnostics_are_disabled_by_default():
    datamodule = FTWMapAfricaSSLDataModule(catalog="unused.csv")
    datamodule.train_dataset = TensorDataset(torch.arange(2))

    assert datamodule.loader_startup_diagnostics is False
    assert type(datamodule.train_dataloader()) is DataLoader


def test_diagnostic_dataloader_preserves_batches(capsys):
    dataset = TensorDataset(torch.arange(6))
    baseline = DataLoader(dataset, batch_size=2)
    diagnostic = DiagnosticDataLoader(
        dataset,
        batch_size=2,
        startup_diagnostics=True,
        startup_log_interval_seconds=60,
    )

    baseline_batches = [batch[0].tolist() for batch in baseline]
    diagnostic_batches = [batch[0].tolist() for batch in diagnostic]

    assert diagnostic_batches == baseline_batches
    output = capsys.readouterr().out
    assert "waiting for first train batch" in output
    assert "first train batch ready" in output


def test_diagnostic_dataloader_stops_heartbeat_after_first_batch(capsys):
    class SlowDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 1

        def __getitem__(self, index):
            time.sleep(0.04)
            return index

    loader = DiagnosticDataLoader(
        SlowDataset(),
        batch_size=1,
        startup_diagnostics=True,
        startup_log_interval_seconds=0.01,
    )

    assert list(loader)[0].tolist() == [0]
    output = capsys.readouterr().out
    assert "still waiting for first train batch" in output

    time.sleep(0.03)
    assert capsys.readouterr().out == ""


def test_dataset_reports_only_first_image_read(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "usage,image\n"
        "train,2017/train_a.tif\n"
        "train,2017/train_b.tif\n"
    )

    def fake_load_image(*args, **kwargs):
        return np.zeros((4, 2, 2), dtype=np.float32)

    monkeypatch.setattr("ftw_ma.ssl_datamodule.load_image", fake_load_image)
    dataset = FTWMapAfricaSSL(
        catalog=str(catalog),
        data_dir=str(tmp_path),
        split="train",
        split_column="usage",
        img_path_cols=["image"],
        loader_startup_diagnostics=True,
    )

    dataset[0]
    dataset[1]

    output = capsys.readouterr().out
    assert output.count("reading first image") == 1
    assert output.count("first image ready") == 1


def test_dataset_skips_bad_image_when_enabled(monkeypatch, tmp_path, capsys):
    catalog = tmp_path / "catalog.csv"
    catalog.write_text(
        "usage,image\n"
        "train,2017/bad.tif\n"
        "train,2017/good.tif\n"
    )

    def fake_load_image(path, *args, **kwargs):
        if "bad.tif" in str(path):
            raise OSError("cannot read test image")
        return np.ones((4, 2, 2), dtype=np.float32)

    monkeypatch.setattr("ftw_ma.ssl_datamodule.load_image", fake_load_image)
    dataset = FTWMapAfricaSSL(
        catalog=str(catalog),
        data_dir=str(tmp_path),
        split="train",
        split_column="usage",
        img_path_cols=["image"],
        skip_bad_images=True,
        max_image_read_retries=1,
    )

    sample = dataset[0]

    assert torch.equal(sample["image"], torch.ones(4, 2, 2))
    output = capsys.readouterr().out
    assert "skipping bad image sample" in output
    assert "bad.tif" in output


def test_dataset_raises_bad_image_when_skip_disabled(monkeypatch, tmp_path):
    catalog = tmp_path / "catalog.csv"
    catalog.write_text("usage,image\ntrain,2017/bad.tif\n")

    def fake_load_image(*args, **kwargs):
        raise OSError("cannot read test image")

    monkeypatch.setattr("ftw_ma.ssl_datamodule.load_image", fake_load_image)
    dataset = FTWMapAfricaSSL(
        catalog=str(catalog),
        data_dir=str(tmp_path),
        split="train",
        split_column="usage",
        img_path_cols=["image"],
        skip_bad_images=False,
    )

    try:
        dataset[0]
    except OSError as error:
        assert "cannot read test image" in str(error)
    else:
        raise AssertionError("Expected bad image read to raise")
