import logging
from typing import Callable

import numpy as np

from src.binning import make_probability_bins, to_bin_type


def make_calibration_function(
    confidences: np.ndarray,
    true_positives: np.ndarray,
    calibration: str = "baseline",
    nr_bins: int = 100,
) -> Callable[[float], float]:
    def calibration_fn(conf):
        return conf

    true_positives = true_positives.astype(float)
    assert len(confidences) == len(true_positives)
    if any(bin_t in calibration for bin_t in ["quantile", "baseline"]):
        bins = make_probability_bins(nr_bins, to_bin_type(calibration), confidences)
        ind_bin = np.digitize(confidences, bins) - 1
        bin_sums = np.bincount(
            ind_bin, weights=confidences, minlength=len(bins)
        ).astype(float)
        bin_true = np.bincount(
            ind_bin, weights=true_positives, minlength=len(bins)
        ).astype(float)
        bin_total = np.bincount(ind_bin, minlength=len(bins)).astype(float)

        correction = np.divide(
            bin_true,
            bin_total,
            out=np.zeros_like(bin_true, dtype=float),
            where=bin_total != 0,
        )
        if "monotonic" in calibration:
            correction = np.maximum.accumulate(correction)

        if "indexing" in calibration:
            # original histogram binning

            def calibration_fn(conf):
                ind_conf_bin = np.digitize(conf, bins) - 1
                return correction[ind_conf_bin]

        elif "spline" in calibration:

            if "centered" in calibration:
                # center of bin
                center_corrections = 0.5 * (bins[:-1] + bins[1:])
                center_corrections = np.concatenate(
                    (center_corrections, [bins[-1]])
                )  # padding, will be removed
            else:
                # weighted by the average within the bin
                center_corrections = np.divide(
                    bin_sums,
                    bin_total,
                    out=np.zeros_like(bin_sums, dtype=float),
                    where=bin_total != 0,
                )
            if "bound" in calibration:
                # no extrapolation
                correction = correction[:-1]
                center_corrections = center_corrections[:-1]
            else:
                # add 0,0 and 1,1 as bounds
                correction = np.concatenate(([0], correction))
                center_corrections = np.concatenate(([0], center_corrections))
                correction[-1], center_corrections[-1] = 1, 1

            assert len(center_corrections) == len(correction)

            def calibration_fn(conf):
                return np.interp(conf, center_corrections, correction)

    else:
        if calibration != "none":
            raise ValueError(f"{calibration} not a valid calibration!")

    return calibration_fn


def is_valid_calibration(calibration_function: Callable[[float], float]) -> bool:
    x_conf = np.linspace(0, 1, 200)
    cal_results = np.vectorize(calibration_function)(x_conf)
    return (
        np.all(np.diff(cal_results) >= 0)
        and min(cal_results) >= 0
        and max(cal_results) <= 1
    )
