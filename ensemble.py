import logging
from ctypes import Union

import hydra
import numpy as np
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
import ensemble_boxes
import pathlib
from pycocotools.coco import COCO
import io
from contextlib import redirect_stdout
from src.utils.general import save_dict, save_dict_as_table, ensure_list

from src.utils.coco_helpers import (
    evaluate_coco,
    coco_dets_to_np,
    np_to_coco_dets,
    load_anns_safe,
)

combine_functions = {
    "nms": ensemble_boxes.ensemble_boxes_nms.nms
}


def combine_boxes(boxes: list[list], image: dict, config: DictConfig) -> list[dict]:
    if not any(boxes):
        return list()
    boxes_np = coco_dets_to_np(boxes, image)
    with redirect_stdout(io.StringIO()):
        merged_boxes = combine_functions[config["nms"]["run_type"]](
            *boxes_np, **config["nms"]["settings"]
        )
    return np_to_coco_dets(merged_boxes, image)


log = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="ensemble")
def ensemble(cfg: DictConfig) -> dict[str, Union[np.ndarray, float]]:
    results = list()
    path_to_annos = pathlib.Path(
        to_absolute_path(cfg["paths"]["annotations"]["path"]),
        cfg["paths"]["annotations"]["split"][cfg["split"]],
    )
    path_to_dets = pathlib.Path(to_absolute_path(cfg["paths"]["detections"]["path"]))

    # load target coco
    with redirect_stdout(io.StringIO()):
        coco_gt = COCO(str(path_to_annos))

    # load coco detection
    det_paths = [
        path_to_dets.joinpath(cfg["detections"][detector][cfg["split"]])
        for detector in ensure_list(cfg["detectors"])
    ]
    det_paths = map(load_anns_safe, det_paths)
    with redirect_stdout(io.StringIO()):
        coco_dets = list(map(coco_gt.loadRes, det_paths))

    for detector, coco_det in zip(cfg["detectors"], coco_dets):
        result = evaluate_coco(coco_gt, coco_det)
        result = {
            metric: v for metric, v in result.items() if "per_class" not in metric
        }
        results.append({**result, **{"det": detector}})

    new_coco_dets = list()
    # combine predictions for each image
    for img_id, image in coco_gt.imgs.items():
        dets = [coco_det.imgToAnns[img_id] for coco_det in coco_dets]
        combined_dets = combine_boxes(dets, image, cfg)
        new_coco_dets.extend(combined_dets)

    with redirect_stdout(io.StringIO()):
        new_coco_dets = coco_gt.loadRes(new_coco_dets)

    # evaluate performance
    result = evaluate_coco(coco_gt, new_coco_dets)
    result = {metric: v for metric, v in result.items() if "per_class" not in metric}
    results.append({**result, **{"det": "ensemble"}})
    log.info(result)

    save_dict_as_table(results, ignore_cols=["mAR", "mAP75"], name="ensemble_eval")

    if cfg["save_outputs"]:
        logging.debug(f"Saving results.")
        save_dict(new_coco_dets.dataset["annotations"], "predictions.json")

    return result


if __name__ == "__main__":
    ensemble()
