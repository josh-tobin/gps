/*
This header is used to define various parameters of sample data that are shared
between the C++ and Python code.
*/
#pragma once

/* TODO: find a way to have just one enum and automatically create the Python wrapper */

// List of data types.
enum DataType
{
    Action = 0,
    JointAngle,
    JointVelocity,
    EndEffectorPoint,
    EndEffectorPoint_velocity,
    EndEffectorPosition,
    EndEffectorRotation,
    EndEffectorJacobian,
    RBGImage,
    TotalDataTypes
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
    enum_<DataType>("data_type")
        .value("action", Action)
        .value("joint_angle", JointAngle)
        .value("joint_velocity", JointVelocity)
        .value("end_effector_point", EndEffectorPoint)
        .value("end_effector_point_velocity", EndEffectorPointVelocity)
        .value("end_effector_position", EndEffectorPosition)
        .value("end_effector_rotation", EndEffectorRotation)
        .value("end_effector_jacobian", EndEffectorJacobian)
        .value("rbg_image",RBGImage)
        .value("total_data_types", TotalDataTypes)
        ;
}

#endif
