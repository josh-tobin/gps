/*
This header is used to define various parameters of sample data that are shared
between the C++ and Python code.
*/
#pragma once

// List of data types.
enum DataType
{
    Actions = 0,
    JointAngles,
    JointVelocities,
    EndEffectorPoints,
    EndEffectorVelocities,
    EndEffectorPosition,
    EndEffectorRotation,
    EndEffectorJacobian,
    RGBImage,
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
        .value("Actions",Actions)
        .value("JointAngles",JointAngles)
        .value("JointVelocities",JointVelocities)
        .value("EndEffectorPoints",EndEffectorPoints)
        .value("EndEffectorVelocities",EndEffectorVelocities)
        .value("EndEffectorPosition",EndEffectorPosition)
        .value("EndEffectorRotation",EndEffectorRotation)
        .value("EndEffectorJacobian",EndEffectorJacobian)
        .value("RGBImage",RGBImage)
        .value("TotalDataTypes",TotalDataTypes)
        ;
}

#endif
