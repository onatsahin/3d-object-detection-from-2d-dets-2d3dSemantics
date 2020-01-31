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
import argparse

class_dict = {'table': 0, 'chair': 1, 'sofa': 2, 'bookcase': 3, 'board': 4}

'''def get_object_list(sem_folder, json_list, class_dict):
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
    return object_indexes'''

def get_gt_center(path):
    with open(path) as f:
        c = f.read().strip().split()
    return float(c[0]), float(c[1]), float(c[2])

def calculate_dist(p1, p2):
    return np.sqrt(((p1[0] - p2[0])**2) + ((p1[0] - p2[0])**2) + ((p1[0] - p2[0])**2))

def row_calc(row):
    det_point = (row['x'], row['y'], row['z'])
    sem_file = '_'.join(row['2d_Detection_File'].split('_')[:-1]) + '_semantic.png'
    np_sem = np.array(Image.open(os.path.join(semantic_dir, sem_file)))
    #print(sem_file)

    bbox_center = (row['y_min'] + row['y_max']) // 2, (row['x_min'] + row['x_max']) // 2
    bbox_upper_left = (row['y_min'] + bbox_center[0]) // 2, (row['x_min'] + bbox_center[1]) // 2
    bbox_upper_right = (row['y_min'] + bbox_center[0]) // 2, (row['x_max'] + bbox_center[1]) // 2
    bbox_lower_left = (row['y_max'] + bbox_center[0]) // 2, (row['x_min'] + bbox_center[1]) // 2
    bbox_lower_right = (bbox_center[0] + row['y_max']) // 2, (bbox_center[1] + row['x_max']) // 2
    #print(bbox_center, bbox_upper_left, bbox_lower_right)
    #print(row['x_min'], row['y_min'], row['x_max'], row['y_max'])
    #print(bbox_center)
    #print('row class', row['Class'])
    indexes = {}
    index_center = utils.get_index(np_sem[int(bbox_center[0]), int(bbox_center[1]), :])
    if index_center not in list(indexes.keys()):
        indexes[index_center] = 1
    else:
        indexes[index_center] += 1

    index_ul = utils.get_index(np_sem[int(bbox_upper_left[0]), int(bbox_upper_left[1]), :])
    if index_ul not in list(indexes.keys()):
        indexes[index_ul] = 1
    else:
        indexes[index_ul] += 1

    index_ur = utils.get_index(np_sem[int(bbox_upper_right[0]), int(bbox_upper_right[1]), :])
    if index_ur not in list(indexes.keys()):
        indexes[index_ur] = 1
    else:
        indexes[index_ur] += 1

    index_ll = utils.get_index(np_sem[int(bbox_lower_left[0]), int(bbox_lower_left[1]), :])
    if index_ll not in list(indexes.keys()):
        indexes[index_ll] = 1
    else:
        indexes[index_ll] += 1

    index_lr = utils.get_index(np_sem[int(bbox_center[0]), int(bbox_center[1]), :])
    if index_lr not in list(indexes.keys()):
        indexes[index_lr] = 1
    else:
        indexes[index_lr] += 1

    selected_index = max(indexes.items(), key=operator.itemgetter(1))[0]
    #print('index_center ', index_center, 'index_ul', index_ul, 'index_lr', index_lr)
    if selected_index >= len(labels) or selected_index <= 0:
        return -1

    selected_label = utils.parse_label(labels[selected_index])
    #label_ul = utils.parse_label(labels[index_ul])
    #label_lr = utils.parse_label(labels[index_lr])
    #print('label_center ', index_center, 'label_ul', index_ul, 'label_lr', index_lr)
    #if selected_label['instance_class'] != row['Class']:
    #    return -1
    #print(label_center['instance_class'], '\n\n')

    center_gt_file = os.path.join(gt_dir, '{}_{}'.format(selected_label['room_type'], selected_label['room_num']), '{}_{}.txt'.format(selected_label['instance_class'], selected_label['instance_num']))
    try:
        gt_center = get_gt_center(center_gt_file)
    except FileNotFoundError:
        print('File Not Found: {}'.format(center_gt_file))
        return
    gt_class = selected_label['instance_class']
    distance = calculate_dist(det_point, gt_center)

    if distance < pred_det_distances[selected_index] and distance < 2:
        pred_det_distances[selected_index] = distance
    #else:
    #    pred_labels[selected_index]
    
def calc_object_internal_distance(object_df):
    x = object_df['x']['max'] - object_df['x']['min']
    y = object_df['y']['max'] - object_df['y']['min']
    z = object_df['z']['max'] - object_df['z']['min']
    
    return np.sqrt((x**2) + (y**2) + (z**2))

def count_classes_from_index_list(index_list, labels):
    result_dict = {}
    for index in index_list:
        obj_info = utils.parse_label(labels[index])
        obj_class = obj_info['instance_class']

        if obj_class not in list(result_dict.keys()):
            result_dict[obj_class] = 1
        else:
            result_dict[obj_class] += 1
    return result_dict

def write_item_dict(item_dict, fp):
    for key, value in item_dict.items():
        fp.write('{}: {}\n'.format(key, value))
    fp.write('\n')
        
parser = argparse.ArgumentParser()
parser.add_argument('--area_dir', type=str, help='2D-3D-S area directory path.')
parser.add_argument('--center_gt_dir', type=str, help='Stanford3dDataset avg_annots_renamed directory path.')
parser.add_argument('--detection_csv', type=str, help='Path of the CSV file that includes 3D mappings.')
#parser.add_argument('--test_area_jsons', nargs='*', type=str, default=[])
parser.add_argument('--object_size_info_json', type=str, help='Path of the area object info json file.')
parser.add_argument('--area_object_index_json', type=str, help='Path of the area test object index json file.')
parser.add_argument('--result_text', type=str, help='Final result txt file.')




args = parser.parse_args()
labels = utils.load_labels('/mnt/data/2D-3D-Semantics/assets/semantic_labels.json')
detections = pd.read_csv(args.detection_csv).dropna()
semantic_dir = os.path.join(args.area_dir, 'data/semantic')
gt_dir = args.center_gt_dir
#test_area_jsons = args.test_area_jsons 

with open(args.object_size_info_json) as f:
    area_object_size_info = json.load(f)

#test_objects = np.array(get_object_list(semantic_dir, test_area_jsons, class_dict))
with open(args.area_object_index_json) as f:
    test_object_indexes = np.array(json.load(f))
    
print('Test objects acquired.')

test_object_indices = np.where(test_object_indexes == True)[0]
#print(test_object_indices)
test_object_not_indices = np.where(test_object_indexes == False)[0]
pred_det_distances = [9999] * 9816
detections.apply(row_calc, axis=1)
print('Prediction distances calculated.')

pred_det_distances = np.array(pred_det_distances)
not_indices_dists = pred_det_distances[test_object_not_indices]
indices_dists = pred_det_distances[test_object_indices]

#print('Object count:', test_object_indices.shape[0])

tp = []
fp_false_obj = []
fp_dist_above_th = []
fn = []

#tp_cd = {}
#fp_false_obj_cd = {}
#fp_dist_above_th_cd = {}
#fn_cd = {}

for i in range(len(pred_det_distances)):
    #print(labels[i])
    if i in test_object_indices:
        if pred_det_distances[i] == 9999:
            fn.append(i)
        #print(pred_labels[i])
        elif pred_det_distances[i] <= calc_object_internal_distance(area_object_size_info[labels[i]]['info_df']) / 2:
            tp.append(i)
        else:
            fp_dist_above_th.append(i)
    else:
        if pred_det_distances[i] != 9999:
            fp_false_obj.append(i)
   
with open(args.result_text, 'a') as results:
    results.write('Object count: {}\n\n'.format(test_object_indices.shape[0]))
    fn_cd = count_classes_from_index_list(fn, labels)
    #print('False negatives:', indices_dists[indices_dists == 9999].shape[0])
    results.write('FALSE NEGATIVES: {}\n'.format(len(fn)))
    write_item_dict(fn_cd, results)

    fp_false_obj_cd = count_classes_from_index_list(fp_false_obj, labels)
    results.write('FALSE POSITIVES (FALSE OBJECT): {}\n'.format(len(fp_false_obj)))
    write_item_dict(fp_false_obj_cd, results)

    fp_dist_above_th_cd = count_classes_from_index_list(fp_dist_above_th, labels)
    results.write('FALSE POSITIVES (DISTANCE ABOVE THRESHOLD): {}\n'.format(len(fp_dist_above_th)))
    write_item_dict(fp_dist_above_th_cd, results)

    tp_cd = count_classes_from_index_list(tp, labels)
    results.write('TRUE POSITIVES: {}\n'.format(len(tp)))
    write_item_dict(tp_cd, results)



'''#false_negative_indices = []


false_positive_object = not_indices_labels[not_indices_labels != 9999]
false_object_classes = {}

for obj in false_positive_object:
    
    

print('False positives (False object):', false_positive_object.shape[0])
false_object_classes = {}

with open(args.object_info_json) as f:
    area_objects = json.load(f)

tp = 0

for index in test_object_indices:
    if pred_det_distances[index] != 9999:
        #print(area_objects[labels[index]])
        if pred_det_distances[index] <= calc_object_internal_distance(area_objects[labels[index]]['info_df']) / 2:# / 2:
            tp += 1

fp_distance = test_object_indices.shape[0] - tp - pred_labels[test_object_indices][pred_labels[test_object_indices] == 9999].shape[0]

print('False positive(distance above th):', fp_distance)
print('True positives:', tp)'''













