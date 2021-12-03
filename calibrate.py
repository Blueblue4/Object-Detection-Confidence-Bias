from typing import Union, Any

import hydra
import numpy as np
from collections import defaultdict
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from pycocotools.coco import COCO
import logging
import pathlib
from sklearn.model_selection import train_test_split
from contextlib import redirect_stdout
import io

from src.calib import make_calibration_object, CalibrationObject
from src.utils.coco_helpers import (
    evaluate_coco,
    dataset_from_image_ids,
    extract_labeled_detections,
    evaluate_calibration,
)
from src.utils.general import append_dict_entries, save_dict_as_table, make_logger

log = make_logger(__name__, log_exceptions=True)


@hydra.main(config_path="config", config_name="default_calibrate")
def calibration_sweep_main(cfg: DictConfig) -> list[dict[str, Any]]:

    annos = cfg["paths"]["annotations"]
    path_to_annos = pathlib.Path(to_absolute_path(annos["path"]))
    annos_path = {
        split: str(path_to_annos.joinpath(split_path))
        for split, split_path in annos["split"].items()
    }
    path_to_dets = pathlib.Path(to_absolute_path(cfg["paths"]["detections"]["path"]))

    all_results = list()

    do_not_need_bins = ["BetaCalibration", "none"]
    do_not_need_box_bins = ["none"]
    calib_split = cfg["calib"]["on_split"]
    oracle_split = cfg["optim"]["oracle_on"]
    gt_det = load_gt_detections(annos_path, cfg)

    predictions_path = cfg["detections"][cfg["detector"]]

    preds = load_model_predictions(
        cfg, gt_det, cfg["detector"], path_to_dets, predictions_path
    )

    if oracle_split:
        stats_baseline_oracle = evaluate_coco(gt_det[oracle_split], preds[oracle_split])

    detections = extract_labeled_detections(
        preds[calib_split], gt_det[calib_split], iou_threshold=cfg["iou_threshold"]
    )

    for calibration in cfg["calib"]["list_types"]:
        logging.info(calibration)
        sweep_results = list()
        for nr_box_bins in cfg["calib"]["box_bins"]:
            if (
                calibration in do_not_need_box_bins
                and nr_box_bins != cfg["calib"]["box_bins"][0]
            ):
                continue
            box_sweep_results = list()
            for nr_bins in cfg["calib"]["conf_bins"]:

                if (
                    calibration in do_not_need_bins
                    and nr_bins != cfg["calib"]["conf_bins"][0]
                ):
                    continue

                results = defaultdict(list)

                calibration_obj, metrics = make_calibration_object(
                    detections,
                    calibration=calibration,
                    nr_bins=nr_bins,
                    bbox_split="quantile",
                    nr_box_bins=nr_box_bins,
                    k_folds=cfg["optim"]["k_folds"],
                    seed=cfg["seed"],
                )

                stats_eval_split = evaluate_calibration(
                    calibration_obj, calib_split, gt_det, preds
                )
                metrics.update({"mAP_calib": stats_eval_split["mAP_per_class"]})
                if oracle_split:
                    stats_cal_oracle = evaluate_calibration(
                        calibration_obj, oracle_split, gt_det, preds
                    )
                    results = append_dict_entries(
                        results, stats_cal_oracle, prefix=oracle_split + "_cal_"
                    )

                results = append_dict_entries(results, metrics, prefix="metric_")
                box_sweep_results.append(
                    {**results.copy(), "calibration_obj": calibration_obj}
                )

            sweep_results.append(box_sweep_results)

        results_dict = {
            metric: np.array(
                [[val[metric] for val in val_list] for val_list in sweep_results]
            )
            for metric in results.keys()
        }

        oracle = dict()
        if oracle_split:
            oracle = eval_oracle(oracle_split, results_dict, stats_baseline_oracle)

        for optimized_metric in list(cfg["optim"]["metrics"]):
            calibration_obj_opt = get_optimized_calibration_object(
                sweep_results, optimized_metric, results_dict, cfg
            )
            results_out = dict()
            for target in list(cfg["apply_on"]):
                if target not in preds:
                    with redirect_stdout(io.StringIO()):
                        if target not in gt_det:
                            gt_det[target] = COCO(annos_path[target])
                        preds[target] = gt_det[target].loadRes(
                            str(path_to_dets.joinpath(predictions_path[target]))
                        )
                save_path = (
                    pathlib.Path(
                        f"{calibration}_on{calib_split}" f"_opt_{optimized_metric}",
                        target,
                        cfg["detector"] + ".json",
                    )
                    if cfg["save_outputs"]
                    else None
                )
                stats_cal_results = evaluate_calibration(
                    calibration_obj_opt, target, gt_det, preds, save_name=save_path
                )

                results_out.update(
                    {
                        f"{target}_cal_{key}": val
                        for key, val in stats_cal_results.items()
                    }
                )
            results_out.update(
                {
                    "calib type": calibration,
                    "model name": cfg["detector"],
                    "optimized metric": optimized_metric,
                    "cal on split": calib_split,
                }
            )
            results_out.update(oracle)
            all_results.append(results_out)

    save_dict_as_table(
        all_results, ignore_cols=["per_class", "mAR", "mAP75"], name="calibration_eval"
    )
    return all_results


def load_gt_detections(annos_path: Union[DictConfig, dict], cfg: DictConfig) -> dict[str, COCO]:
    logging.debug("Loading gt annotations")
    gt_det = dict()
    with redirect_stdout(io.StringIO()):
        gt_det["val"] = COCO(annos_path["val"])
        if cfg["calib"]["on_split"] == "train":
            gt_det["train"] = COCO(annos_path["train"])
        elif cfg["calib"]["on_split"] == "val_split1":
            split1_img_ids, split2_img_ids = train_test_split(
                list(gt_det["val"].imgs.keys()),
                train_size=cfg["calib"]["split_size"],
                random_state=cfg["seed"],
            )
            gt_det["val_split1"] = dataset_from_image_ids(split1_img_ids, gt_det["val"])
            gt_det["val_split2"] = dataset_from_image_ids(split2_img_ids, gt_det["val"])
    return gt_det


def load_model_predictions(
    cfg: DictConfig,
    gt_det: dict[str, COCO],
    model_name: str,
    path_to_dets: pathlib.Path,
    predictions_path: dict[str, str],
) -> dict[str, COCO]:
    logging.debug(f"Loading predictions model: {model_name}")
    preds = dict()
    with redirect_stdout(io.StringIO()):
        preds["val"] = gt_det["val"].loadRes(
            str(path_to_dets.joinpath(predictions_path["val"]))
        )
        if cfg["calib"]["on_split"] == "train":
            preds["train"] = gt_det["train"].loadRes(
                str(path_to_dets.joinpath(predictions_path["train"]))
            )
        elif cfg["calib"]["on_split"] == "val_split1":
            preds["val_split1"] = dataset_from_image_ids(
                list(gt_det["val_split1"].imgs.keys()), preds["val"]
            )
            preds["val_split2"] = dataset_from_image_ids(
                list(gt_det["val_split2"].imgs.keys()), preds["val"]
            )
    return preds


def eval_oracle(
    split: str, results_dict: dict[str, np.ndarray], stats_baseline_oracle: dict
) -> dict[str, np.ndarray]:
    nr_classes = results_dict[split + "_cal_mAP_per_class"].shape[-1]
    max_index_class = np.unravel_index(
        np.argmax(
            np.reshape(results_dict[split + "_cal_mAP_per_class"], (-1, nr_classes)),
            axis=0,
        ),
        results_dict[split + "_cal_mAP"].shape,
    )
    logging.info(
        f"Max index mAP (oracle): box splits{max_index_class[0]}, conf {max_index_class[1]}"
    )
    oracle = dict()
    for met in ["mAP", "mAP50"]:
        oracle["baseline_" + met] = stats_baseline_oracle[met]
        max_met = np.mean(
            np.max(
                np.reshape(
                    results_dict[f"{split}_cal_{met}_per_class"], (-1, nr_classes)
                ),
                axis=0,
            )
        )
        logging.info(f"theoretical max {met}: {max_met:.7f} ")
        oracle["theoretical_max_" + met] = max_met
    return oracle


def get_optimized_calibration_object(
    sweep_results: list[list[dict[str, Union[np.ndarray, CalibrationObject]]]],
    optimized_metric: str,
    results_dict: dict[str, np.ndarray],
    cfg: DictConfig,
) -> CalibrationObject:
    logging.debug(f"Combining calibration object from best {optimized_metric}")
    calibration_obj_opt = dict()
    if optimized_metric == "tradeoff_full":
        results_dict["metric_" + optimized_metric] = (
            np.max(results_dict["metric_mean_diff_all"], axis=0)
            - results_dict["metric_mean_diff_all"]
        ) ** 2 + results_dict["metric_true_var"]
    if optimized_metric == "oracle":
        results_dict["metric_oracle"] = -results_dict[
            f"{cfg['optim']['oracle_on']}_cal_mAP_per_class"
        ]
    opt_results = results_dict["metric_" + optimized_metric]
    index_optimized = np.unravel_index(
        np.argmin(np.reshape(opt_results, (-1, opt_results.shape[-1])), axis=0),
        opt_results.shape[:-1],
    )
    logging.info(
        f"min indexes for {optimized_metric}: box splits{index_optimized[0]}, conf {index_optimized[1]}"
    )
    for indx, cls in enumerate(sweep_results[0][0]["calibration_obj"].keys()):
        calibration_obj_opt[cls] = sweep_results[index_optimized[0][indx]][
            index_optimized[1][indx]
        ]["calibration_obj"][cls]
    return calibration_obj_opt


if __name__ == "__main__":
    calibration_sweep_main()
