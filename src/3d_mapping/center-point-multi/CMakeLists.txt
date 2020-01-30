cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

project(range_image_creation)

find_package(PCL 1.2 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (convert_coordinates convert_coordinates.cpp)
target_link_libraries (convert_coordinates ${PCL_LIBRARIES})
