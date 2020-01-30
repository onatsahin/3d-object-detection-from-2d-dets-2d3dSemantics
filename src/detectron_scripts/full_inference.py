import os
import numpy as np
import json
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.structures import BoxMode
import itertools
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
import cv2
import glob
from PIL import Image, ImageDraw
import sys
sys.path.append("/mnt/data/2D-3D-Semantics")
import assets.utils as utils
import matplotlib.pyplot as plt
import random
import argparse
from subprocess import Popen, PIPE

class_dict = {'table': 0, 'chair': 1, 'sofa': 2, 'bookcase': 3, 'board': 4}
class_list = ['table', 'chair', 'sofa', 'bookcase', 'board']

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

parser = argparse.ArgumentParser()
parser.add_argument('--area_dir', type=str)
parser.add_argument('--area_json', type=str)
#parser.add_argument('--pointcloud', type=str)
parser.add_argument('--model_path', type=str)
parser.add_argument('--detection_dir', type=str)

args = parser.parse_args()

cfg = get_cfg()
cfg.merge_from_file("/home/ubuntu/detectron2_repo/configs/COCO-Detection/retinanet_R_101_FPN_3x.yaml")
cfg.DATALOADER.NUM_WORKERS = 2
#cfg.MODEL.WEIGHTS = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"  # initialize from model zoo
cfg.TEST.EXPECTED_RESULTS = [['bbox', 'AP', 38.5, 0.2]]
cfg.TEST.EVAL_PERIOD = 5
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.MAX_ITER = 1000000000    # 300 iterations seems good enough, but you can certainly train longer
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5  # only has one class (ballon)
cfg.MODEL.WEIGHTS = args.model_path

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

img_dir = os.path.join(args.area_dir, 'data', 'rgb')
pose_dir = os.path.join(args.area_dir, 'data', 'pose')

dataset_dicts = get_stanford_dicts(img_dir, args.area_json)

for d in dataset_dicts:
    
    img_base_path = ('_').join(d['file_name'].split('_')[:-1])
    pose_file_path = os.path.join(pose_dir, img_base_path + '_pose.json')
    img_file_path  = d['file_name']  
    det_file_name = "{}.txt".format(os.path.basename(d['file_name']).split('.')[0])
    det_file_path = os.path.join(args.detection_dir, 'detections', det_file_name)
    gt_file_path = os.path.join(args.detection_dir, 'groundtruths', det_file_name)
    os.system( 'touch {}'.format(det_file_path) )
    
    for ann in d['annotations']:
        line = "{} {} {} {} {}".format( class_list[ann['category_id']], ann['bbox'][0], ann['bbox'][1], ann['bbox'][2], ann['bbox'][3] )
        #print("echo '{}' >> {}".format(line, gt_file_path))
        os.system("echo {} >> {}".format(line, gt_file_path))
    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    
    detections = outputs["instances"].to("cpu").get_fields()
    detection_boxes = detections['pred_boxes'].tensor
    detection_classes = detections['pred_classes']
    detection_scores = detections['scores']
    detection_num = detection_boxes.shape[0]
    
    for i in range(detection_num):
        line = "{} {} {} {} {} {}".format( class_list[detection_classes[i].item()], detection_scores[i], detection_boxes[i][0], detection_boxes[i][1], detection_boxes[i][2], detection_boxes[i][3] )
        #print(line)
        os.system("echo {} >> {}".format(line, det_file_path))
    
    
    
    
  
