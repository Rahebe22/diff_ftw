import numpy as np

from ftw_ma.normalize import normalize_image


def test_normalize_image_replaces_nan_inf_and_nodata_with_zero():
    image = np.array(
        [
            [
                [0.0, 1.0],
                [np.nan, np.inf],
            ],
            [
                [2.0, 65535.0],
                [-1.0, 4.0],
            ],
        ],
        dtype=np.float32,
    )

    normalized = normalize_image(
        image,
        strategy="min_max",
        procedure="lab",
        nodata=[0, 65535],
    )

    assert normalized.dtype == np.float32
    assert np.isfinite(normalized).all()
    assert normalized[0, 0, 0] == 0
    assert normalized[0, 1, 0] == 0
    assert normalized[0, 1, 1] == 0
    assert normalized[1, 0, 1] == 0
    assert normalized[1, 1, 0] == 0


def test_normalize_image_all_invalid_chip_returns_zeros():
    image = np.array(
        [
            [[np.nan, np.inf]],
            [[-1.0, 65535.0]],
        ],
        dtype=np.float32,
    )

    normalized = normalize_image(
        image,
        strategy="min_max",
        procedure="lab",
        nodata=[65535],
    )

    assert normalized.dtype == np.float32
    assert np.array_equal(normalized, np.zeros_like(image, dtype=np.float32))


def test_normalize_image_empty_nodata_still_masks_nonfinite_values():
    image = np.array(
        [
            [[1.0, np.nan]],
            [[2.0, np.inf]],
        ],
        dtype=np.float32,
    )

    normalized = normalize_image(
        image,
        strategy="min_max",
        procedure="lab",
        nodata=[],
    )

    assert np.isfinite(normalized).all()
    assert normalized[0, 0, 1] == 0
    assert normalized[1, 0, 1] == 0
