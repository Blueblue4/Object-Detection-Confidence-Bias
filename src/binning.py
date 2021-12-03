import collections.abc
from typing import Any, Union, Optional

import numpy as np


def make_probability_bins(
    nr_bins: int,
    bin_type: str = "linear",
    data_points: Optional[Union[list, np.ndarray]] = None,
) -> np.ndarray:
    if nr_bins > len(data_points):
        nr_bins = len(data_points)
    if nr_bins < 1:
        return np.array([0, 1])

    if bin_type == "linear":
        bins = np.linspace(0.0, 1 + 1e-8, nr_bins + 1)
    elif bin_type == "quantile":
        quantiles = np.linspace(0, 1, nr_bins + 1)
        bins = np.percentile(data_points, quantiles * 100)
        bins[-1] = 1.0 + 1e-8
        bins[0] = 0
        # make monotonic and remove duplicates
        bins = np.unique(np.maximum.accumulate(bins))
    else:
        raise ValueError(f"{bin_type}, not a valid binning type for probabilities")
    return bins


def make_size_bins(
    nr_bins: int,
    bin_type: str = "linear",
    data_points: Optional[Union[list, np.ndarray]] = None,
) -> np.ndarray:
    if nr_bins > len(data_points):
        nr_bins = len(data_points)
    if nr_bins < 1:
        return np.array([0, 1e5 ** 2])

    if bin_type == "linear":
        bins = np.linspace(0.0, max(data_points) + 1e-8, nr_bins + 1)
    elif bin_type == "quantile":
        quantiles = np.linspace(0, 1, nr_bins + 1)
        bins = np.percentile(data_points, quantiles * 100)
        bins[-1] = bins[-1] + 1e-8
        bins[0] = 0
        # make monotonic and remove duplicates
        bins = np.unique(np.maximum.accumulate(bins))
    elif bin_type == "default":
        bins = np.array([0, 32 ** 2, 96 ** 2, 1e5 ** 2])  # coco standard
    else:
        raise ValueError(f"{bin_type}, not a valid binning type")
    return bins


class BinningDict(dict):
    def __init__(
        self,
        bins: np.ndarray,
        default_value: Any = None,
    ):
        super().__init__()
        self.bins = bins
        self.default_value = default_value

    def __missing__(self, key: Any) -> Optional[float]:
        if isinstance(key, collections.abc.Sequence):
            assert len(key) == 1
            key = key[0]
        key_clipped = np.clip(
            key, a_min=min(self.bins) + 1e-8, a_max=max(self.bins) - 1e-8
        )
        ind_bin = self.get_bin_index(key_clipped)
        assert ind_bin >= 0
        return self.get(ind_bin, self.default_value)

    def get_bin_index(self, key: Any) -> np.ndarray:
        return np.digitize(key, self.bins) - 1


def to_bin_type(calib: str) -> str:
    if "baseline" in calib:
        return "linear"
    elif "quantile" in calib:
        return "quantile"
    elif "default" in calib:
        return "default"
    else:
        raise ValueError(
            f"No binning type defined in calibration-string {calib}. \
        Options are linear, quantile or default!"
        )
