import json
import glob
from PIL import Image, ImageDraw
import pandas as pd
import sys
import os
sys.path.append("/mnt/data/2D-3D-Semantics")
import assets.utils as utils
import numpy as np
import operator
import matplotlib.pyplot as plt
import json

class_dict = {'table': 0, 'chair': 1, 'sofa': 2, 'bookcase': 3, 'board': 4}

def save_object_list(sem_folder, json_list, class_dict, output_json):
    anns = utils.load_labels('/mnt/data/2D-3D-Semantics/assets/semantic_labels.json')
    object_indexes = [False] * len(anns)

    for jf in json_list:
        with open(jf) as f:
            annot_dict = json.load(f)
            objects = list(annot_dict.keys())
            #print(jf, len(objects))
            for obj in objects:
                #print('img')
                sem_file_name = "_".join(obj.split('_')[:-1] + ['semantic.png'])
                sem_file_path = os.path.join(sem_folder, sem_file_name)

                sem_img = np.array(Image.open(sem_file_path))

                obj_colors = np.unique(sem_img.reshape(-1, sem_img.shape[2]), axis=0)
                for color in obj_colors:
                    index = utils.get_index(color)
                    if index > len(anns):
                        continue
                    lbl = utils.parse_label(anns[index])
                    if lbl['instance_class'] in class_dict.keys():
                        mask = (sem_img == color).all(-1)
                        i, j = np.where(mask)
                        if i.shape[0] < 250:
                            continue

                        object_indexes[index] = True
    
    with open(output_json, 'w') as f:
        json.dump(object_indexes, f)
        
save_object_list('/mnt/data/2D-3D-Semantics/area_3/data/semantic/', ['/home/ubuntu/stanford-detectron/area_json_files/area_3_partitioned/area_3_test.json', '/home/ubuntu/stanford-detectron/area_json_files/area_3_partitioned/area_3_val.json'], class_dict, 'area_3_test_object_indexes.json')
print('Area 3 complete')

save_object_list('/mnt/data/2D-3D-Semantics/area_1/data/semantic/', ['/home/ubuntu/stanford-detectron/area_json_files/area_1_partitioned/area_1_test.json'], class_dict, 'area_1_test_object_indexes.json')
print('Area 1 complete')

save_object_list('/mnt/data/2D-3D-Semantics/area_2/data/semantic/', ['/home/ubuntu/stanford-detectron/area_json_files/area_2_partitioned/area2_test.json'], class_dict, 'area_2_test_object_indexes.json')
print('Area 2 complete')

save_object_list('/mnt/data/2D-3D-Semantics/area_4/data/semantic/', ['/home/ubuntu/stanford-detectron/area_json_files/area_4_partitioned/area_4_test.json'], class_dict, 'area_4_test_object_indexes.json')
print('Area 4 complete')












