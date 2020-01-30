#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/io/png_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/common/float_image_utils.h>
#include <pcl/point_types.h>
#include <pcl/impl/point_types.hpp>
#include <cmath>
#include <string>
#include <iostream>
#include <sstream>
#include <cstdlib>

using namespace std;

void pointPickingEventOccurred (const pcl::visualization::PointPickingEvent& event, void* viewer_void);
void visualize_and_mark(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, vector< vector<float> > &coordinates);
vector<float> get_3d_coordinates_from_range_image(pcl::RangeImagePlanar &range_image, float range_image_x, float range_image_y);

int main (int argc, char** argv) {

  float x = atof(argv[2]);
  float y = atof(argv[3]);
  float z = atof(argv[4]);
  float roll = atof(argv[5]);
  float pitch = atof(argv[6]);
  float yaw = atof(argv[7]);
  float range_image_width = atof(argv[8]);
  float range_image_height = atof(argv[9]);
  float range_image_center_x = range_image_width / 2;
  float range_image_center_y = range_image_height / 2;
  float noiseLevel= atof(argv[10]);//0.00;
  float minRange = atof(argv[11]);//0.2f;
  int borderSize = atoi(argv[12]);//0;
  float focal_length_x = atof(argv[13]);//637.5;
  float focal_length_y = atof(argv[14]);//637.5;
  char* range_img_name = argv[15];
  //float coordinates_2d_x = atof(argv[16])/4;
  //float coordinates_2d_y = atof(argv[17])/4;

  pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud (new pcl::PointCloud<pcl::PointXYZ>);
  pcl::PCDReader reader;
  //pcl::io::loadPCDFile("../../realsense_meshes/16_11_18/2_post_processed/cloud.pcd", *pointCloud);
  //pcl::io::loadPCDFile(argv[1], *pointCloud);
  pcl::io::loadPLYFile(argv[1], *pointCloud);
  Eigen::Affine3f sensorPose = pcl::getTransformation(x, y, z, roll, pitch, yaw);
  pcl::RangeImagePlanar::CoordinateFrame coordinate_frame = pcl::RangeImagePlanar::LASER_FRAME;
  
  pcl::RangeImagePlanar rangeImage, rangeImage_half_res, rangeImage_quarter_res;
  rangeImage.createFromPointCloudWithFixedSize(*pointCloud, range_image_width, range_image_height, range_image_center_x, range_image_center_y, focal_length_x, focal_length_y, sensorPose, coordinate_frame, noiseLevel, minRange); //image size
  rangeImage.getHalfImage(rangeImage_half_res); //QUARTER
  rangeImage_half_res.getHalfImage(rangeImage_quarter_res);

  float* ranges = rangeImage_quarter_res.getRangesArray(); //QUARTER
  unsigned char* rgb_image = pcl::visualization::FloatImageUtils::getVisualImage (ranges, rangeImage_quarter_res.width, rangeImage_quarter_res.height); 

  pcl::io::saveRgbPNGFile(range_img_name, rgb_image, rangeImage_quarter_res.width, rangeImage_quarter_res.height); 

  //float* ranges = rangeImage.getRangesArray();
  //unsigned char* rgb_image = pcl::visualization::FloatImageUtils::getVisualImage (ranges, rangeImage.width, rangeImage.height); 
  //pcl::io::saveRgbPNGFile(range_img_name, rgb_image, rangeImage.width, rangeImage.height);
 
  vector< vector<float> > coordinates_3d;

  //coordinates_3d.push_back(get_3d_coordinates_from_range_image(rangeImage_quarter_res, to_get_x, to_get_y));

  for(int i=16; i<argc; i+=2){
      //cout << atof(argv[i])/4 << ' ' << atof(argv[i+1])/4 << endl;
      vector<float> xyz_coordinates = get_3d_coordinates_from_range_image(rangeImage_quarter_res, atof(argv[i])/4, atof(argv[i+1])/4);
  
      cout << xyz_coordinates[0] << ' ' << xyz_coordinates[1] << ' ' << xyz_coordinates[2] << endl;
      //break;
  }
  /*vector<float> xyz_coordinates = get_3d_coordinates_from_range_image(rangeImage_quarter_res, coordinates_2d_x, coordinates_2d_y);
  
  cout << xyz_coordinates[0] << ' ' << xyz_coordinates[1] << ' ' << xyz_coordinates[2] << endl;*/
  //VISUALIZATION
  //visualize_and_mark(pointCloud, coordinates_3d);



  return 0;
}

void pointPickingEventOccurred (const pcl::visualization::PointPickingEvent& event, void* viewer_void)
{
  std::cout << "[INOF] Point picking event occurred." << std::endl;

  float x, y, z;
  if (event.getPointIndex () == -1)
  {
     return;
  }
  event.getPoint(x, y, z);
  std::cout << "[INFO] Point coordinate ( " << x << ", " << y << ", " << z << ")" << std::endl;
}

void visualize_and_mark(pcl::PointCloud<pcl::PointXYZRGBA>::Ptr &cloud, vector< vector<float> > &coordinates){
  pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
  viewer.addPointCloud(cloud,"body");
  viewer.registerPointPickingCallback (pointPickingEventOccurred, (void*)&viewer);

  for(int i=0; i<coordinates.size(); i++){
    pcl::ModelCoefficients sphere_coeff;
    sphere_coeff.values.resize (4);
    
    for(int j=0; j<3; j++){
      sphere_coeff.values[j] = coordinates[i][j];
    }

    sphere_coeff.values[3] = 0.1;
    viewer.addSphere(sphere_coeff, "sphere");
  }
  viewer.spin();
}

vector<float> get_3d_coordinates_from_range_image(pcl::RangeImagePlanar &range_image, float range_image_x, float range_image_y){
  std::string PointWithRange_data, xyz_coordinates_str, temp;
  stringstream buffer, buffer2;
  int xyz_length;
  float x_3d, y_3d, z_3d;
  vector<float> xyz_coordinates;
 
  buffer << range_image.getPoint(range_image_x, range_image_y);
  PointWithRange_data = buffer.str();

  //cout << PointWithRange_data << endl;
  //cout << "3d: " << rangeImage_quarter_res.getPoint(to_get_x2, to_get_y2) << endl;
  xyz_length = PointWithRange_data.find(' ') - 1;
  xyz_coordinates_str = PointWithRange_data.substr(1, xyz_length);
  //cout << xyz_coordinates_str << endl;

  buffer.flush();
  buffer2 << xyz_coordinates_str;

  while(getline(buffer2, temp, ','))
    xyz_coordinates.push_back(atof(temp.c_str()));
  
  return xyz_coordinates;
}
