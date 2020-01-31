# Mapping Object Detections from RGB Images to 3D Using Point Clouds

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

*