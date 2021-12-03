import logging
from collections import defaultdict
from typing import Dict

import numpy as np
from sklearn.metrics import average_precision_score, log_loss
from sklearn.model_selection import KFold

from src.binning import BinningDict, make_size_bins
from src.calibration_factory import make_calibration_function
from src.utils.coco_helpers import LabeledDetections

CalibrationObject = Dict[int, BinningDict]


def make_calibration_object(
    detections: LabeledDetections,
    calibration="baseline",
    nr_bins=100,
    bbox_split="quantile",
    nr_box_bins=100,
    k_folds=1,
    seed=1,
    save_labels=False,
) -> CalibrationObject:
    calibration_object = dict()
    metrics = defaultdict(list)

    for category_id, detection in detections.items():
        binned_detections = BinningDict(
            bins=make_size_bins(nr_box_bins, bin_type=bbox_split, data_points=detection["box_size"]),
            default_value=list(),
        )

        calib_func_all = np.vectorize(
            make_calibration_function(
                detection["confidences"],
                detection["true_positives"],
                calibration=calibration,
                nr_bins=nr_bins,
            )
        )
        diff_calib_bin_to_all = list()

        confs_fold, confs_all_fold, tps_fold, confs_bin = list(), list(), list(), list()
        bin_numbers = binned_detections.get_bin_index(detection["box_size"])
        for bin_number in np.unique(bin_numbers):
            bin_mask = bin_numbers == bin_number
            confs_binned = detection["confidences"][bin_mask]
            tps_binned = detection["true_positives"][bin_mask]

            assert len(confs_binned) == len(tps_binned)
            calib_func = make_calibration_function(
                confs_binned, tps_binned, calibration=calibration, nr_bins=nr_bins
            )

            bin_dict = {"calibrate": calib_func}
            if save_labels:
                bin_dict = {
                    **bin_dict,
                    "confidences": confs_binned,
                    "true_positives": tps_binned,
                }
            binned_detections[bin_number] = bin_dict

            # the rest is eval metrics: move to separate function?
            cal_confs_binned = np.vectorize(calib_func)(confs_binned)
            diff_calib_bin_to_all.append(
                cal_confs_binned - calib_func_all(confs_binned)
            )

            if k_folds > 1 and len(confs_binned) > 1:
                confs_bin_fold = list()
                if len(confs_binned) < k_folds:
                    logging.error(
                        f"Number of folds ({k_folds}) is larger than number of "
                        f"data points ({len(confs_binned)}) for cat: {category_id}"
                    )
                    folds = len(confs_binned)
                else:
                    folds = k_folds
                k_fold_splitter = KFold(n_splits=folds, shuffle=True, random_state=seed)
                for train_ind, val_ind in k_fold_splitter.split(confs_binned):
                    calib_func_k = make_calibration_function(
                        confs_binned[train_ind],
                        tps_binned[train_ind],
                        calibration=calibration,
                        nr_bins=nr_bins,
                    )

                    confs_fold.append(np.vectorize(calib_func_k)(confs_binned[val_ind]))
                    confs_bin_fold.append(np.vectorize(calib_func_k)(confs_binned))
                    tps_fold.append(tps_binned[val_ind])
                    confs_bin.append(cal_confs_binned[val_ind])
                confs_all_fold.append(np.stack(confs_bin_fold, axis=1))
        if k_folds > 1 and len(confs_fold) > 1:
            metric = get_metrics(
                confs_bin, confs_fold, confs_all_fold, diff_calib_bin_to_all, tps_fold
            )
        else:
            metric = default_metrics()
        for met, val in metric.items():
            metrics[met].append(val)
        calibration_object[category_id] = binned_detections

    return calibration_object, metrics


def expected_calibration_error(confidences, true_positives, nr_bins=10):
    expected_precision = np.vectorize(
        make_calibration_function(
            confidences,
            true_positives,
            calibration="baseline_indexing",
            nr_bins=nr_bins,
        )
    )
    return np.mean(np.abs(expected_precision(confidences) - confidences))


def get_metrics(
    confidences_bin,
    confidences_fold,
    confidences_all_fold,
    diff_calib_bin_to_all,
    true_positives_fold,
):
    diff_calib_bin_to_all = np.concatenate(diff_calib_bin_to_all, axis=0)
    variances_all_fold = np.concatenate(
        [np.var(fold, axis=1) for fold in confidences_all_fold], axis=0
    )
    confidences_fold = np.concatenate(confidences_fold, axis=0)
    true_positives_fold = np.concatenate(true_positives_fold, axis=0)
    confidences_bin = np.concatenate(confidences_bin, axis=0)
    k_fold_diff_tp = confidences_fold - true_positives_fold
    variation_k_fold = confidences_bin - confidences_fold
    metric = {
        "mean_diff_all": np.abs(diff_calib_bin_to_all).mean(),
        "mse_all": (diff_calib_bin_to_all ** 2).mean(),
        "k_fold_abs_deviation": np.abs(variation_k_fold).mean(),
        "k_fold_var": (variation_k_fold ** 2).mean(),
        "true_var": variances_all_fold.mean(),
        "k_fold_brier": (k_fold_diff_tp ** 2).mean(),
        "k_fold_log": log_loss(
            y_true=true_positives_fold, y_pred=confidences_fold, labels=[True, False]
        ),
        "k_fold_diff": np.abs(k_fold_diff_tp).mean(),
        "tradeoff_b2v": -np.abs(diff_calib_bin_to_all).mean() ** 2
        + variances_all_fold.mean(),
        "tradeoff_bv": -np.abs(diff_calib_bin_to_all).mean()
        + (variation_k_fold ** 2).mean(),
        "ece": expected_calibration_error(confidences_fold, true_positives_fold),
        "ap_estimate": -average_precision_score(true_positives_fold, confidences_fold)
        if any(true_positives_fold)
        else 0.0,
    }
    return metric


def default_metrics():
    return {"mean_diff_all": 1.,
            "mse_all": 1.,
            "k_fold_abs_deviation": 1.,
            "k_fold_var": 1.,
            "true_var": 1.,
            "k_fold_brier": 1.,
            "k_fold_log": np.inf,
            "k_fold_diff": 1.,
            "tradeoff_b2v": 1.,
            "tradeoff_bv": 1.,
            "ece": 1.,
            "ap_estimate": 0.}
