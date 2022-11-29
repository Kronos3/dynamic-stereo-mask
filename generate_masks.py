from pathlib import Path
import sys

import numpy as np
import cv2

from tqdm import tqdm

from src import kitti
from src.lea_stereo import LEAStereo

import torch
import detectron2
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import _PanopticPrediction
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
setup_logger()

torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(0.5, 0)

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)

dataset = kitti.KittiDataset(Path('dataset/sequences/00'))

SKY_ID = 40

all_clouds = []
for index in tqdm(range(int(sys.argv[1]), int(sys.argv[2]))):
    frame = kitti.KittiFrame(dataset, index)

    cv2.imwrite("left/%06d.png"% index, frame.left_color())
    cv2.imwrite("right/%06d.png"% index, frame.right_color())

    panoptic_seg, segments_info = predictor(frame.left_color())["panoptic_seg"]
    pred = _PanopticPrediction(
        panoptic_seg.to("cpu"),
        segments_info,
        MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))

    overall_mask = np.ones(frame.left_color().shape[0:2])
    for mask, sinfo in pred.semantic_masks():
        category_idx = sinfo["category_id"]
        if category_idx == SKY_ID:
            continue

        # Semantic masks are good (stuff)
        overall_mask[mask] = 0

    np.save("masks/mask%d" % index, overall_mask)
