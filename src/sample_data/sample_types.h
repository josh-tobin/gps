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
    EndEffectorPointVelocity,
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
    enum_<DataType>("DataType")
        .value("Action", Action)
        .value("JointAngle", JointAngle)
        .value("JointVelocity", JointVelocity)
        .value("EndEffectorPoint", EndEffectorPoint)
        .value("EndEffectorPointVelocity", EndEffectorPointVelocity)
        .value("EndEffectorPosition", EndEffectorPosition)
        .value("EndEffectorRotation", EndEffectorRotation)
        .value("EndEffectorJacobian", EndEffectorJacobian)
        .value("RBGImage",RBGImage)
        .value("TotalDataTypes", TotalDataTypes)
        ;
}

#endif
