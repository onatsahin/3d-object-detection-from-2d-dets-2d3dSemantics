import argparse
import os
import glob
import json
import pandas as pd
from subprocess import Popen, PIPE

parser = argparse.ArgumentParser()
parser.add_argument('--pointcloud', type=str, help='Path of the area point cloud.')
parser.add_argument('--detection_dir', type=str, help='Path of the directory that contains 2D detection and ground truth files.')
parser.add_argument('--area_dir', type=str, help='2D-3D-S area directory path.')
parser.add_argument('--output_csv', type=str, help='Output CSV file to be created that includes 3D mappings.')

args = parser.parse_args()
detection_files = glob.glob(os.path.join(args.detection_dir, 'detections', '*txt'))

results = pd.DataFrame(columns=['2d_Detection_File', 'Class', 'Confidence', 'x_min', 'y_min', 'x_max', 'y_max', 'x', 'y', 'z'])
header = True
print('Number of Detections: ', len(detection_files))
for i, d in enumerate(detection_files):
    print('{}/{}  {}'.format(i, len(detection_files), d))
    with open(d) as d_file:
        dets = d_file.read().split('\n')[:-1]

    for i in range(len(dets)):
        dets[i] = dets[i].split(' ')

        for j in range(1, 6):
            dets[i][j] = float(dets[i][j])
    #print(dets)
    pose_file_name = ('_').join(os.path.basename(d).split('_')[:-1]) + '_pose.json'
    pose_file_path = os.path.join(args.area_dir, 'data/pose', pose_file_name)
    
    with open(pose_file_path) as p_file:
        pose = json.load(p_file)
        
    x = pose['camera_location'][0]
    y = pose['camera_location'][1]
    z = pose['camera_location'][2]
    roll = pose['final_camera_rotation'][1]
    pitch = 1.57 - pose['final_camera_rotation'][0]
    yaw = 1.57 + pose['final_camera_rotation'][2]
    focal_length = pose['camera_k_matrix'][0][0]
    
    #print(x, y, z, roll, pitch, yaw, focal_length)
    center_points = []
    
    '''for det in dets:
        center_point_x = ((det[2] + det[4]) / 2) / 4
        center_point_y = ((det[3] + det[5]) / 2) / 4
        print('\n\n\n')
        print(det)
        
        args_2d_to_3d = ['/home/ubuntu/stanford-detectron/pcl_mapping/center-point/convert_coordinates', args.pointcloud, str(x), str(y), str(z), str(roll), str(pitch), str(yaw), str(center_point_x), str(center_point_y), '1080', '1080', '0', '0.2', '0', str(focal_length), str(focal_length), 'range.png']
        print(center_point_x, center_point_y)
        p = Popen(args_2d_to_3d, stdout=PIPE, stderr=PIPE)
        output, err = p.communicate()
        point = [float(j) for j in output.split()]
        print(output)
        print(point)'''
    
    for det in dets:
        center_points.append(str((det[2] + det[4]) / 2))
        center_points.append(str((det[3] + det[5]) / 2))
        
    
    args_2d_to_3d = ['center-point-multi/convert_coordinates', args.pointcloud, str(x), str(y), str(z), str(roll), str(pitch), str(yaw), '1080', '1080', '0', '0.2', '0', str(focal_length), str(focal_length), 'range.png'] + center_points
    
    #print(args_2d_to_3d)
    p = Popen(args_2d_to_3d, stdout=PIPE, stderr=PIPE)
    output = p.stdout.read()
    center_points = [cp.split() for cp in output.split(b'\n')[:-1]]
    #print(center_points)
    #print(err)
    
    for i in range(len(center_points)):
        results = results.append({'2d_Detection_File': os.path.basename(d), 'Class': dets[i][0], 'Confidence': float(dets[i][1]), 'x_min': float(dets[i][2]), 'y_min': float(dets[i][3]), 'x_max': float(dets[i][4]), 'y_max': float(dets[i][5]), 'x': float(center_points[i][0]), 'y': float(center_points[i][1]), 'z': float(center_points[i][2])}, ignore_index=True)
        
        if len(results.index) == 50:
            results = results.astype({'Confidence': float, 'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float, 'x': float, 'y': float, 'z': float})
            results.to_csv(args.output_csv, index=False, header=header, mode = 'a')
            header = False
            results = results.iloc[0:0]
    
    
results = results.astype({'Confidence': float, 'x_min': float, 'y_min': float, 'x_max': float, 'y_max': float, 'x': float, 'y': float, 'z': float})
results.to_csv(args.output_csv, index=False, header=header, mode = 'a')

            
            
            
                
