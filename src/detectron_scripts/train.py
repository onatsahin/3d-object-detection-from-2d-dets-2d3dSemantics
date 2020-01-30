import os
import numpy as np
import json
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.structures import BoxMode
import itertools
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import cv2
import glob
from PIL import Image, ImageDraw
import sys
sys.path.append("/mnt/data/2D-3D-Semantics")
import assets.utils as utils
import matplotlib.pyplot as plt

class_dict = {'table': 0, 'chair': 1, 'sofa': 2, 'bookcase': 3, 'board': 4}

def get_stanford_dicts(img_dir, json_path):
    with open(json_path) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for _, v in imgs_anns.items():
        record = {}

        filename = os.path.join(img_dir, v["file_name"])
        record["file_name"] = filename
        record["height"] = v["height"]
        record["width"] = v["width"]

        annos = v["objects"]
        objs = []
        for _, anno in annos.items():
            obj = {
                "bbox": anno['bbox'],
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": anno['category_id'],
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

'''
for d in ['1', '2', '3', '4', '6']:
    print('/home/ubuntu/stanford-detectron/area_json_files/area_{}.json'.format(d))
    DatasetCatalog.register("2d_3d_semantics_area_{}".format(d), lambda l=d: get_stanford_dicts('/mnt/data/2D-3D-Semantics/area_{}/data/rgb/'.format(d), '/home/ubuntu/stanford-detectron/area_json_files/area_{}.json'.format(d)))
    MetadataCatalog.get("2d_3d_semantics_area_{}".format(d)).set(thing_classes=['table', 'chair', 'sofa', 'bookcase', 'board'])
'''
print('/home/ubuntu/stanford-detectron/area_json_files/area_4_partitioned/area_4_train.json')
DatasetCatalog.register("2d_3d_semantics_area4_train", lambda l='/mnt/data/2D-3D-Semantics/area_4/data/rgb/': get_stanford_dicts(l, '/home/ubuntu/stanford-detectron/area_json_files/area_4_partitioned/area_4_train.json'))
MetadataCatalog.get("2d_3d_semantics_area4_train").set(thing_classes=['table', 'chair', 'sofa', 'bookcase', 'board'])
stanford_metadata = MetadataCatalog.get("2d_3d_semantics_area4_train")

#DatasetCatalog.register("2d_3d_semantics_area_3_train", lambda l='/home/ubuntu/2D-3D-Semantics/area_3/data/rgb/': get_stanford_dicts(l, '/home/ubuntu/2D-3D-Semantics/area_3/data/area_3_train.json'))
#MetadataCatalog.get("2d_3d_semantics_area_3_train").set(thing_classes=['table', 'chair', 'sofa', 'bookcase', 'board'])
#stanford_metadata = MetadataCatalog.get("2d_3d_semantics_area_3_train")

cfg = get_cfg()
#cfg.merge_from_file("/home/ubuntu/detectron2_repo/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
#cfg.merge_from_file("/home/ubuntu/detectron2_repo/configs/COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml")
cfg.merge_from_file("/home/ubuntu/detectron2_repo/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml")
#cfg.DATASETS.TRAIN = (["2d_3d_semantics_area_{}".format(d) for d in ['1', '2', '3', '4', '6']])
cfg.DATASETS.TRAIN = (["2d_3d_semantics_area4_train"])
print(cfg.DATASETS.TRAIN)
cfg.DATASETS.TEST = ()   # no metrics implemented for this dataset
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # initialize from model zoo
#cfg.MODEL.WEIGHTS = "/home/ubuntu/stanford-detectron/out_faster_rcnn_R_101_FPN_3x_1-3-5-6/model_0074999.pth"
#cfg.MODEL.WEIGHTS = "/home/ubuntu/stanford-detectron/output_train_1-3-5-6/model_0124999.pth"
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_101_FPN_3x/137851257/model_final_f6e8b1.pkl"
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/retinanet_R_101_FPN_3x/138363263/model_final_59f53c.pkl"
cfg.MODEL.WEIGHTS = "/home/ubuntu/stanford-detectron/out_retinanet_R_101_FPN_3x_4train_lr001/model_0099999_weights.pth"
#cfg.TEST.EXPECTED_RESULTS = [['bbox', 'AP', 38.5, 0.2]]
#cfg.TEST.EVAL_PERIOD = 5
cfg.SOLVER.IMS_PER_BATCH = 8
cfg.SOLVER.BASE_LR = 0.0001
#cfg.SOLVER.WARMUP_FACTOR = 1.0 / 100
cfg.SOLVER.WARMUP_ITERS = 0
#cfg.SOLVER.STEPS = ([130030])
#cfg.SOLVER.GAMMA = 0.1
cfg.SOLVER.MAX_ITER = 1000000000    # 300 iterations seems good enough, but you can certainly train longer
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32   # faster, and good enough for this toy dataset
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon)
cfg.OUTPUT_DIR = "./out_retinanet_R_101_FPN_3x_4train_lr0001"
#cfg.OUTPUT_DIR = './out_faster_rcnn_R_101_FPN_3x_1-3-5-6_part2'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
print(trainer.train())
