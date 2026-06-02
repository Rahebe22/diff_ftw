from scripts.build_ssl_catalog_cache import build_cache
from ftw_ma.ssl_datamodule import (
    ConstantMemoryDistributedSampler,
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
