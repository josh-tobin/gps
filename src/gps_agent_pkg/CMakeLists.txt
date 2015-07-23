cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})

# Find PCL 1.7
#find_package(PCL 1.7 REQUIRED)
#include_directories(BEFORE ${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})
#if(NOT PCL_FOUND)
#    MESSAGE(FATAL_ERROR "PCL not found.\n")
#endif (NOT PCL_FOUND)

# List the files to be included for DDP controller.
#set(DDP_FILES src/ddp_controller.cpp
#			  src/kinematic_controller.cpp
#              src/kinematic_controller_pos.cpp
#		      src/torque_controller_zoh.cpp
#		      src/torque_controller_foh.cpp
#		      src/torque_controller_jt.cpp
#		      src/state_estimator.cpp
#		      src/neural_network.cpp
#		      src/reset_controller.cpp)

# Boost Python
FIND_PACKAGE(Boost 1.45.0)
IF(Boost_FOUND)
  INCLUDE_DIRECTORIES("${Boost_INCLUDE_DIRS}" "/usr/include/python2.7")
  SET(Boost_USE_STATIC_LIBS OFF)
  SET(Boost_USE_MULTITHREADED ON)
  SET(Boost_USE_STATIC_RUNTIME OFF)
  FIND_PACKAGE(Boost 1.45.0 COMPONENTS python)
  #ADD_LIBRARY(yay SHARED yay.cxx)
  #TARGET_LINK_LIBRARIES(yay ${Boost_LIBRARIES})
ELSEIF(NOT Boost_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find correct Boost version. Did you set BOOST_ROOT?")
ENDIF()

IF(CMAKE_COMPILER_IS_GNUCXX)
  ADD_DEFINITIONS("-Wall")
ELSE()
  MESSAGE(FATAL_ERROR "CMakeLists.txt has not been tested/written for your compiler.")
ENDIF()

set(DDP_FILES src/robotplugin.cpp
			  src/pr2plugin.cpp
              src/sample.cpp
		      src/sensor.cpp
		      src/lingausscontroller.cpp
		      src/positioncontroller.cpp
		      src/trialcontroller.cpp
		      src/encodersensor.cpp
		      src/encoderfilter.cpp)

# Include Caffe
if (USE_CAFFE)
    # add definitions for the C++ code
    add_definitions(-DUSE_CAFFE)
    if (USE_CAFFE_GPU)
        add_definitions(-DCAFFE_GPU)
        find_package(CUDA 5.5 REQUIRED)
        include_directories(${CUDA_INCLUDE_DIRS})
    else (USE_CAFFE_GPU)
        add_definitions(-DCPU_ONLY)
    endif (USE_CAFFE_GPU)
    # add paths to CAFFE
    link_directories(${CAFFE_LIBRARY_PATH})
    include_directories(${CAFFE_INCLUDE_PATH})
    # extra header paths
    include_directories(${CAFFE_EXTRA_PATH})
    # add neural network to DDP controller files
    #set(DDP_FILES ${DDP_FILES} src/neural_network_caffe.cpp)
    # compile Caffe test
    #rosbuild_add_executable(caffe_test src/caffe_test.cpp src/neural_network_caffe.cpp)
    #target_link_libraries(caffe_test caffe protobuf)
    # compile image processor node
    #rosbuild_add_executable(caffe_img_processor src/img_processor.cpp src/neural_network_caffe.cpp)
    #target_link_libraries(caffe_img_processor caffe protobuf)
endif (USE_CAFFE)

rosbuild_add_library(ddp_controller_lib ${DDP_FILES})

#rosbuild_add_executable(kinematic_baseline src/kinematic_baseline.cpp)

#rosbuild_add_executable(controller_switcher src/controller_switcher.cpp)

#rosbuild_add_executable(pointcloud_solver src/pointcloud_solver.cpp
#                                          src/keypoint_detector.cpp)

#rosbuild_add_executable(point_head src/point_head.cpp)
#rosbuild_add_executable(torso src/torso.cpp)

#SET_TARGET_PROPERTIES(pointcloud_solver PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})

# Include Caffe in controller
if (USE_CAFFE)
    target_link_libraries(ddp_controller_lib caffe protobuf)
endif (USE_CAFFE)

