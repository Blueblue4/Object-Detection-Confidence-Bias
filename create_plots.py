import io
from contextlib import redirect_stdout
import dufte
import numpy as np
import tikzplotlib
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import hydra
from hydra.utils import to_absolute_path
import pathlib

from src.calib import make_calibration_object
from src.calibration_factory import make_calibration_function
from src.utils.coco_helpers import extract_labeled_detections
from src.utils.general import ensure_list

plt.style.use(dufte.style)


@hydra.main(config_path="config", config_name="plot_miscal")
def plot_calibration_confidences(cfg):

    path_to_annos = pathlib.Path(
        to_absolute_path(cfg["paths"]["annotations"]["path"]),
        cfg["paths"]["annotations"]["split"][cfg["split"]],
    )
    path_to_det = pathlib.Path(
        to_absolute_path(cfg["paths"]["detections"]["path"]),
        cfg["detections"][cfg["detector"]][cfg["split"]],
    )

    with redirect_stdout(io.StringIO()):
        gt = COCO(str(path_to_annos))
        pred = gt.loadRes(str(path_to_det))
    detections = extract_labeled_detections(pred, gt, iou_threshold=0.5)

    calibration_obj, _ = make_calibration_object(
        detections,
        calibration=cfg["calib"]["type"],
        nr_bins=cfg["calib"]["conf_bins"],
        bbox_split=cfg["calib"]["bbox_split"],
        nr_box_bins=cfg["calib"]["box_bins"],
        save_labels=True,
    )

    calibration_obj_all, _ = make_calibration_object(
        detections,
        calibration=cfg["calib"]["type"],
        nr_bins=cfg["calib"]["conf_bins"],
        bbox_split="linear",
        nr_box_bins=1,
    )

    for cal_class in ensure_list(cfg["classes"]):
        plt.plot(
            [0, 1], [0, 1], "--", label="perfect calibration", c=(0.5, 0.5, 0.5, 0.8)
        )

        size_borders = [
            f"{size_b:.1f}" for size_b in calibration_obj[cal_class].bins
        ] + ["max"]
        x_conf = np.linspace(0, 1, 200)
        for box_quant, cal_dict in calibration_obj[cal_class].items():

            nr_dets = len(cal_dict["confidences"])
            all_calib_funcs = list()
            # bootstrap
            for i in range(cfg["bootstrap_samples"]):
                sample_ids = np.random.choice(np.arange(nr_dets), nr_dets, replace=True)
                calib_func = make_calibration_function(
                    cal_dict["confidences"][sample_ids],
                    cal_dict["true_positives"][sample_ids],
                    calibration=cfg["calib"]["type"],
                    nr_bins=cfg["calib"]["conf_bins"],
                )
                all_calib_funcs.append(np.vectorize(calib_func)(x_conf))
            recal_func = np.vectorize(cal_dict["calibrate"])
            plt.plot(
                x_conf,
                recal_func(x_conf),
                label=f" {size_borders[box_quant]} - {size_borders[box_quant+1]}",
                alpha=0.9,
            )
            all_calib_funcs = np.stack(all_calib_funcs)
            plt.fill_between(
                x_conf,
                np.percentile(all_calib_funcs, q=2.5, axis=0),
                np.percentile(all_calib_funcs, q=97.5, axis=0),
                alpha=0.1,
            )

        plt.plot(
            x_conf,
            np.vectorize(calibration_obj_all[cal_class][0]["calibrate"])(x_conf),
            label=f"all",
            c=(0.3, 0.3, 0.3),
            alpha=0.9,
        )
        plt.legend()
        plt.title(
            f"{cfg['split']} {cfg['calib']['type']}-Calib curve class:{cal_class}, by bbox size"
        )
        plt.ylabel("Precision")
        plt.xlabel("Confidence")
        plt.axis([0, 1, 0, 1.0])
        filename = f"conf_{cfg['calib']['type']}_Calib_curve_class_{cal_class}{cfg['split']}{cfg['name']}"
        plt.tight_layout()
        plt.savefig(filename + ".svg")
        tikzplotlib.save(filename + ".tex")
        plt.show()


if __name__ == "__main__":
    plot_calibration_confidences()
