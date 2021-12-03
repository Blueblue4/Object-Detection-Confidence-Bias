import io
import json
import logging
import pathlib
from collections import defaultdict
from contextlib import redirect_stdout
from copy import deepcopy
from typing import Union, Optional, Dict

import numpy as np
from pycocotools.coco import COCO
from sklearn.model_selection import train_test_split
from src.utils.general import save_dict, DisableLoggerContext

log = logging.getLogger(__name__)
from pycocotools.cocoeval import COCOeval

try:
    import torch
    from fast_coco_eval import COCOeval_fast
except ImportError:
    COCOeval_fast = COCOeval
    log.info(
        "Failed to import COCOeval_fast with dependency torch! Using slower COCOeval."
    )

LabeledDetections = Dict[int, dict[str, np.ndarray]]


def dataset_from_image_ids(img_ids: list, coco_ds: COCO) -> COCO:
    with redirect_stdout(io.StringIO()):
        new_coco_ds = COCO(None)
        anns = coco_ds.loadAnns(ids=coco_ds.getAnnIds(imgIds=img_ids))
        new_coco_ds.dataset = {
            "categories": coco_ds.dataset["categories"],
            "annotations": anns,
            "images": coco_ds.loadImgs(img_ids),
        }
        new_coco_ds.createIndex()
    return new_coco_ds


def evaluate_coco(coco_gt: COCO, coco_pred: COCO) -> dict[str, Union[np.ndarray, float]]:
    with redirect_stdout(io.StringIO()), DisableLoggerContext():
        coco_eval = COCOeval_fast(coco_gt, coco_pred, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        maps_per_class = map_per_class(coco_eval)
        map50s_per_class = map_per_class(coco_eval, iou_thr=0.5)
        coco_eval.summarize()
    eval_vals = [metric * 100 for metric in coco_eval.stats]
    eval_cats = [
        "mAP",
        "mAP50",
        "mAP75",
        "mAP_small",
        "mAP_medium",
        "mAP_large",
        "mAR_1",
        "mAR_10",
        "mAR_100",
        "mAR_small",
        "mAR_medium",
        "mAR_large",
    ]
    return {
        **dict(zip(eval_cats, eval_vals)),
        **{"mAP_per_class": maps_per_class, "mAP50_per_class": map50s_per_class},
    }


def map_per_class(
    coco_eval: COCOeval,
    iou_thr: Optional[float] = None,
    area_rng: str = "all",
    max_dets: int = 100,
) -> np.ndarray:
    p = coco_eval.eval["params"]
    aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == area_rng]
    mind = [i for i, mDet in enumerate(p.maxDets) if mDet == max_dets]
    s = coco_eval.eval["precision"]
    maps = list()
    if iou_thr is not None:
        t = np.where(iou_thr == p.iouThrs)[0]
        s = s[t]
    for cat in range(coco_eval.eval["counts"][2]):
        prec = s[:, :, cat, aind, mind]
        maps.append(np.mean(prec[prec > -1]))
    return np.array(maps) * 100.0


def split_train_test(
    coco_pred: COCO, coco_gt: COCO, seed: int = 123, split: float = 0.5
) -> tuple[COCO, COCO, COCO, COCO]:
    train_img_ids, test_img_ids = train_test_split(
        list(coco_gt.imgs.keys()), train_size=split, random_state=seed
    )
    train_pred = dataset_from_image_ids(train_img_ids, coco_pred)
    train_gt = dataset_from_image_ids(train_img_ids, coco_gt)
    test_pred = dataset_from_image_ids(test_img_ids, coco_pred)
    test_gt = dataset_from_image_ids(test_img_ids, coco_gt)
    return train_pred, train_gt, test_pred, test_gt


def extract_labeled_detections(
    pred: COCO, gt: COCO, iou_threshold=0.5
) -> LabeledDetections:
    with redirect_stdout(io.StringIO()):
        coco_eval = COCOeval(gt, pred, iouType="bbox")
        coco_eval.params.iouThrs = [iou_threshold]
        coco_eval.params.maxDets = [500]
        coco_eval.params.areaRng = [[0 ** 2, 1e5 ** 2]]  # all
        coco_eval.evaluate()
    detections = {categ: defaultdict(lambda: list()) for categ in gt.cats}

    # Order detections and their labels by src categories
    for image_eval in [image for image in coco_eval.evalImgs if image is not None]:
        category_id = image_eval["category_id"]

        valid_det = np.logical_not(image_eval["dtIgnore"])
        # need to load detections for actual area
        dets = coco_eval.cocoDt.loadAnns(
            np.array(image_eval["dtIds"])[valid_det.squeeze(0)]
        )
        confidence = np.array([det["score"] for det in dets])
        box_size = np.array([det["area"] for det in dets])
        assert len(confidence.shape) == 1
        detections[category_id]["confidences"].append(confidence)
        detections[category_id]["true_positives"].append(
            image_eval["dtMatches"][valid_det] > 0
        )
        detections[category_id]["box_size"].append(box_size)
    detections = {
        cat_id: {
            attr: np.concatenate(attr_list) for attr, attr_list in attr_dict.items()
        }
        for cat_id, attr_dict in detections.items()
    }

    return detections


def calibrate_coco_predictions(calibrations: dict, coco_predictions: COCO) -> COCO:
    for index, prediction in enumerate(coco_predictions.dataset["annotations"]):
        try:
            calibrated_score = calibrations[prediction["category_id"]][
                prediction["area"]
            ]["calibrate"](prediction["score"])
        except TypeError:
            ind = calibrations[prediction["category_id"]].get_bin_index(
                prediction["area"]
            )
            logging.error(
                f"No Calibration found for area: {prediction['area']}, mapped to index: {ind}, bins: {calibrations[prediction['category_id']].bins}"
            )
            calibrated_score = prediction["score"]
        coco_predictions.dataset["annotations"][index]["score"] = calibrated_score

    with redirect_stdout(io.StringIO()):
        coco_predictions.createIndex()
    return coco_predictions


def evaluate_calibration(
    calibration_obj: dict,
    split: str,
    gt_det: dict[str, COCO],
    preds: dict[str, COCO],
    save_name: Optional[str] = None,
) -> dict[str, Union[np.ndarray, float]]:
    logging.debug(f"Applying calibration to {split}")
    calibrated_preds = calibrate_coco_predictions(
        calibration_obj, deepcopy(preds[split])
    )
    logging.debug(f"Evaluating calibrated {split} set")
    eval_cal = evaluate_coco(gt_det[split], calibrated_preds)

    if save_name is not None:
        logging.debug(f"Saving results to {save_name}")
        save_path = pathlib.Path(save_name)
        save_dict(calibrated_preds.dataset["annotations"], save_path)
        save_path = save_path.with_name("evaluations_" + save_path.name)
        save_dict(eval_cal, save_path)
    del calibrated_preds
    return eval_cal


def coco_dets_to_np(
    detections: list[list[dict]], image: dict
) -> tuple[list[np.ndarray], list, list]:
    boxes, scores, ids = list(), list(), list()
    for detection in detections:
        if not detection:
            continue
        detection = {k: np.array([dic[k] for dic in detection]) for k in detection[0]}
        bbox = detection["bbox"]
        bbox = np.stack(
            [bbox[:, 0], bbox[:, 1], bbox[:, 0] + bbox[:, 2], bbox[:, 1] + bbox[:, 3]],
            axis=1,
        )
        boxes.append(
            bbox
            / np.array(
                [image["width"], image["height"], image["width"], image["height"]],
                dtype=np.float32,
            )
        )
        scores.append(detection["score"]), ids.append(detection["category_id"])
    return boxes, scores, ids


def np_to_coco_dets(detections: tuple, image: dict) -> list[dict]:
    dets = dict()
    bbox, dets["score"], dets["category_id"] = detections
    bbox = np.stack(
        [bbox[:, 0], bbox[:, 1], bbox[:, 2] - bbox[:, 0], bbox[:, 3] - bbox[:, 1]],
        axis=1,
    )
    dets["bbox"] = bbox * np.array(
        [image["width"], image["height"], image["width"], image["height"]],
        dtype=np.float32,
    )
    dets["image_id"] = np.full_like(dets["score"], image["id"], dtype=int)
    return [dict(zip(dets, t)) for t in zip(*dets.values())]


def load_anns_safe(filename: Union[pathlib.Path, str]) -> list:
    filename = (
        pathlib.Path(filename) if not isinstance(filename, pathlib.Path) else filename
    )
    assert (
        filename.is_file()
    ), f"Failed to load anns from {filename}, because the file does not exist!"
    with filename.open("r") as f:
        anns = json.load(f)
    if isinstance(anns, dict):
        anns = anns["annotations"]
    assert isinstance(
        anns, list
    ), f"Failed to load anns from {filename}, the format unknown!"
    return anns
