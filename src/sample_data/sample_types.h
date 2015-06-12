/*
This header is used to define various parameters of sample data that are shared
between the C++ and Python code.
*/
#pragma once

/* TODO: find a way to have just one enum and automatically create the Python wrapper */
// List of data types.
enum data_type
{
    action = 0,
    joint_angle,
    joint_velocity,
    end_effector_point,
    end_effector_point_velocity,
    end_effector_position,
    end_effector_rotation,
    end_effector_jacobian,
    rbg_image,
    total_data_types
};

// Check if this is running on the robot.
#ifndef RUN_ON_ROBOT

// Headers.
#include <boost/python.hpp>

// Use boost python.
using namespace boost::python;

// Python implementation.
BOOST_PYTHON_MODULE(gps_sample_types)
{
    enum_<DataType>("DataType")
        .value("action",action)
        .value("joint_angle",joint_angle)
        .value("joint_velocity",joint_velocity)
        .value("end_effector_point",end_effector_point)
        .value("end_effector_point_velocity",end_effector_point_velocity)
        .value("end_effector_position",end_effector_position)
        .value("end_effector_rotation",end_effector_rotation)
        .value("end_effector_jacobian",end_effector_jacobian)
        .value("rbg_image",RGBImage)
        .value("total_data_types",total_data_types)
        ;
}

#endif
