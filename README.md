# The Box Size Confidence Bias Harms Your Object Detector - Code
[![arxiv](https://img.shields.io/badge/arXiv-cs.CV:2112.01901-B31B1B.svg)](https://arxiv.org/abs/2112.01901)

> **Disclaimer:**
>This repository is for research purposes only. It is designed to maintain reproducibility of the experiments described in "The Box Size Confidence Bias Harms Your Object Detector".

## Setup
### Download Annotations
Download COCO2017 annotations for train, val, and test-dev from [here](https://cocodataset.org/#download)
and move them into the folder structure like this (alternatively change the config in `config/all/paths/annotations/coco_2017.yaml` to your local folder structure): 
```
 .
 └── data
   └── coco
      └── annotations
        ├── instances_train2017.json
        ├── instances_val2017.json
        └── image_info_test-dev2017.json
```

### Generate Detections (or Download them [here](https://github.com/Blueblue4/Object-Detection-Confidence-Bias/releases/tag/v0.1))

Generate detections on the train, val, and test-dev COCO2017 set, save them in the COCO file format as JSON files.
Move detections to `data/detections/MODEL_NAME`, see `config/all/detections/default_all.yaml` for all the used detectors and to add other detectors.  
The official implementations for the used detectors are:

- CenterNet(HG) [[Link]](https://github.com/xingyizhou/CenterNet)
- Faster-RCNN, YOLOv3-320, DETR, RetinaNet, CenterNet (RN-18), etc. [[Link]](https://github.com/open-mmlab/mmdetection)
- SSD [[Link]](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD)
- CenterNet2 [[Link]](https://github.com/xingyizhou/CenterNet2)
- YOLOv5x [[Link]](https://github.com/ultralytics/yolov5)
- EfficientDet D0-7 [[Link]](https://github.com/google/automl/tree/master/efficientdet)

### Examples 
##### CenterNet (Hourglass)
To generate the Detections for CenterNet with Hourglass backbone first follow the [installation instructions](https://github.com/xingyizhou/CenterNet/blob/master/readme/INSTALL.md). Then download `ctdet_coco_hg.pth` to `/models` from the [official source](https://github.com/xingyizhou/CenterNet/blob/master/readme/MODEL_ZOO.md)
Then generate the detections from the `/src` folder:  
```bash
# On val
python3 test.py ctdet --arch hourglass --exp_id Centernet_HG_val --dataset coco --load_model ../models/ctdet_coco_hg.pth 
# On test-dev
python3 test.py ctdet --arch hourglass --exp_id Centernet_HG_test-dev --dataset coco --load_model ../models/ctdet_coco_hg.pth --trainval
# On train
sed '56s/.*/  split = "train"/' test.py > test_train.py
python3 test_train.py ctdet --arch hourglass --exp_id Centernet_HG_train --dataset coco --load_model ../models/ctdet_coco_hg.pth
```

The scaling for TTA is set via the `"--test_scales LIST_SCALES"` flag. So to generate only the 0.5x-scales: `--test_scales 0.5`

##### RetinaNet with MMDetection
To generate the de detection files using mmdet, first follow the [installation instructions](https://github.com/open-mmlab/mmdetection/blob/master/docs/get_started.md). Then download specific model weights, in this example `retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth` to `PATH_TO_DOWNLOADED_WEIGHTS` and execute the following commands:
```bash
python3 tools/test.py configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py PATH_TO_DOWNLOADED_WEIGHTS/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth  --eval bbox --eval-options jsonfile_prefix='PATH_TO_THIS_REPO/detections/retinanet_x101_64x4d_fpn_2x/train2017' --cfg-options data.test.img_prefix='PATH_TO_COCO_IMGS/train2017' data.test.ann_file='PATH_TO_COCO_ANNS/annotations/instances_train2017.json'
python3 tools/test.py configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py PATH_TO_DOWNLOADED_WEIGHTS/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth  --eval bbox --eval-options jsonfile_prefix='PATH_TO_THIS_REPO/detections/retinanet_x101_64x4d_fpn_2x/val2017' --cfg-options data.test.img_prefix='PATH_TO_COCO_IMGS/val2017' data.test.ann_file='PATH_TO_COCO_ANNS/annotations/instances_val2017.json'
python3 tools/test.py configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py PATH_TO_DOWNLOADED_WEIGHTS/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth  --eval bbox --eval-options jsonfile_prefix='PATH_TO_THIS_REPO/detections/retinanet_x101_64x4d_fpn_2x/test-dev2017' --cfg-options data.test.img_prefix='PATH_TO_COCO_IMGS/test2017' data.test.ann_file='PATH_TO_COCO_ANNS/annotations/image_info_test-dev2017.json'
```
### Install Dependencies
```bash
pip3 install -r requirements.txt
```
##### Optional Dependencies
```bash
# Faster coco evaluation (used if available)
pip3 install fast_coco_eval
# Parallel multi-runs, if enough RAM is available (add "hydra/launcher=joblib" to every command with -m flag)
pip install hydra-joblib-launcher
```

## Experiments
Most of the experiments are performed using the CenterNet(HG) detections to change the detector add `detections=OTHER_DETECTOR`, with the location of OTHER_DETECTORs detections specified in `config/all/detections/default_all.yaml`.
The results of each experiment are saved to `outputs/EXPERIMENT/DATE` and `multirun/EXPERIMENT/DATE` in the case of a multirun (-m flag).

#### Figure 2: Calibration curve of histogram binning and modified version
```bash
# original histogram binning calibration curve
python3 create_plots.py -cn plot_org_hist_bin
# modified histogram binning calibration curve:
python3 create_plots.py -cn plot_mod_hist_bin
```

#### Table 1: Ablation  of  histogram  binning  modifications
```bash
python3 calibrate.py -cn ablate_modified_hist 
```

#### Table 2: Ablation of optimization metrics of calibration on validation split
```bash
python3 calibrate.py -cn ablate_metrics  "seed=range(4,14)" -m
```

#### Figure 3: Bounding box size bias on train and val data detections
Plot of calibration curve:
```bash
# on validation data
python3 create_plots.py -cn plot_miscal name="plot_miscal_val" split="val"
# on train data:
python3 create_plots.py -cn plot_miscal name="plot_miscal_train" split="train" calib.conf_bins=20
```

#### Table 3: Ablation of optimization metrics of calibration on training data
```bash
python3 calibrate.py -cn explore_train
```

#### Table 4: Effect of individual calibration on TTA
1. Generate detections (on train and val split) for each scale-factor individually `(CenterNet_HG_TTA_050, CenterNet_HG_TTA_075, CenterNet_HG_TTA_100, CenterNet_HG_TTA_125, CenterNet_HG_TTA_150)` and for complete TTA `(CenterNet_HG_TTA_ens)`

2. Generate individually calibrated detections..
    ```bash
    python3 calibrate.py -cn calibrate_train name="calibrate_train_tta" detector="CenterNet_HG_TTA_050","CenterNet_HG_TTA_075","CenterNet_HG_TTA_100","CenterNet_HG_TTA_125","CenterNet_HG_TTA_150","CenterNet_HG_TTA_ens" -m
    ```
3. Copy calibrated detections from `multirun/calibrate_train_tta/DATE/MODEL_NAME/quantile_spline_ontrain_opt_tradeoff_full/val/MODEL_NAME.json` to `data/calibrated/MODEL_NAME/val/results.json` for `MODEL_NAME` in `(CenterNet_HG_TTA_050, CenterNet_HG_TTA_075, CenterNet_HG_TTA_100, CenterNet_HG_TTA_125, CenterNet_HG_TTA_150)`.
4. Generate TTA of calibrated detections  
    ```bash
    python3 enseble.py -cn enseble
    ```

#### Figure 4: Ablation of IoU threshold
```bash
python3 calibrate.py -cn calibrate_train name="ablate_iou" "iou_threshold=range(0.5,0.96,0.05)" -m
```

#### Table 5: Calibration method on different model
```bash
python3 calibrate.py -cn calibrate_train name="calibrate_all_models" detector=LIST_ALL_MODELS -m
```
The test-dev predictions are found in `multirun/calibrate_all_models/DATE/MODEL_NAME/quantile_spline_ontrain_opt_tradeoff_full/test/MODEL_NAME.json` and can be evaluated using the official [evaluation sever](https://cocodataset.org/#upload).

### Supplementary Material

#### A.Figure 5 & 6: Performance Change for Extended Optimization Metrics
```bash
python3 calibrate.py -cn ablate_metrics_extended  "seed=range(4,14)" -m
```

#### A.Table 6: Influence of parameter search spaces on performance gain
```bash
# Results for B0, C0
python3 calibrate.py -cn calibrate_train
# Results for B0, C1
python3 calibrate.py -cn calibrate_train_larger_cbins
# Results for B0 union B1, C0
python3 calibrate.py -cn calibrate_train_larger_bbins
# Results for B0 union B1, C0 union C1
python3 calibrate.py -cn calibrate_train_larger_cbbins
```

### A.Table 7: Influence of calibration method on different sized versions of EfficientDet
```bash
python3 calibrate.py -cn calibrate_train name="influence_modelsize" detector="Efficientdet_D0","Efficientdet_D1","Efficientdet_D2","Efficientdet_D3","Efficientdet_D4","Efficientdet_D5","Efficientdet_D6","Efficientdet_D7" -m
```


## Citation
```
@article{gilg2021box,
      title={The Box Size Confidence Bias Harms Your Object Detector}, 
      author={Johannes Gilg and Torben Teepe and Fabian Herzog and Gerhard Rigoll},
      year={2021},
      eprint={2112.01901},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```   
