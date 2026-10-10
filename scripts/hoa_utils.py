"""Shared helpers for accessing HOA datasets via hoa-tools."""
from __future__ import annotations

import numpy as np


def get_dataset_array(
    dataset_name: str, downsample_level: int = 0
) -> np.ndarray:
    """Open an HOA dataset as a lazy dask-backed xarray and return the raw
    zarr array view (dims z, y, x) without loading everything into RAM."""
    import hoa_tools.dataset

    ds = hoa_tools.dataset.get_dataset(dataset_name)
    return ds.data_array(downsample_level=downsample_level).data
