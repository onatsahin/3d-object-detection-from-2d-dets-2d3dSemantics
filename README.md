# Mapping Object Detections from RGB Images to 3D Using Point Clouds

For detailed information about the project, used codes and files, refer to the included project report.

## Prerequisites

* Create a conda environment using the included yml file.
```
conda env create -f {path to repository}/detectron2_env.yml
```

* Install Detectron2: https://github.com/facebookresearch/detectron2

* Install Point Cloud Library (PCL):
```
sudo apt-get update 
```
```
sudo apt-get install libpcl-dev
```
```
sudo apt install pcl-tools
```
```
sudo apt-get install libproj-dev
```
```
sudo ln -s /usr/lib/x86_64-linux-gnu/libvtkCommonCore-6.2.so /usr/lib/libvtkproj4.so
```
* Get the 2D-3D-S Dataset and place it to data/2D-3D-Semantics: http://buildingparser.stanford.edu/dataset.html#Download
* Get the S3DIS Dataset and place it to data/Stanford3dDataset_v1.2
* Combine room point clouds for each area to create full area point clouds. This can be done with Meshlab: http://www.meshlab.net/

## Usage

* Use a Detectron2 script to perfom object detection training on area training images. A sample script train.py is included in src/detectron_scripts, but it should be edited for different scenarios.

* Use src/detectron_scripts/full_inference.py to perform inference on an area's test set. Use --help or -h option to get more information about the command line arguments. This code generates detection and ground truth files for each image to detection_dir.
```
python full_inference.py --area_dir {area dir path} --area_json {area json path} --model_path {model path} --model_config_path {model config path} --detection_dir {detection dir path}
```

* It is possible to evaluate the used network model with the previously generated ground truth and detection files using src/evaluation/Object-Detection-Metrtics (https://github.com/rafaelpadilla/Object-Detection-Metrics)
```
python pascalvoc.py --gtfolder {ground truth folder path} --detfolder {detection folder path}
```
* Use src/3d_mapping/full_3d_inference.py to map center points of the detected object bounding boxes' to 3D using the previously generated detection files. This code creates a CSV file that contains the 3D points. Use --help or -h option to get more information about the command line arguments.
```
python full_3d_inference.py --pointcloud {point cloud path} --detection_dir {detection path} --area_dir {area directory path} --output_csv {output CSV path}
```
* Use src/evalutation/3d_center_eval.py to evaluate the 3d center points mapped from bounding boxes. This evaluation uses the distance between the mapped point and the object's true center point in the point cloud. The code generates a text file that includes the results. Use --help or -h option to get more information about the command line arguments. 
```
python 3d_center_eval.py --area_dir {area dir path} --center_gt_dir {Stanford3dDataset avg_annots_renamed directory path} --detection_csv {3d detection CSV path} --object_size_info_json {Path of the area object info json file} --area_object_index_json {Path of the area test object index json file} --result_text {Result text file path}
```
