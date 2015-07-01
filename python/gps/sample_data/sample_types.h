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
    JointAngles,
    JointVelocities,
    EndEffectorPoints,
    EndEffectorPointVelocities,
    EndEffectorPositions,
    EndEffectorRotations,
    EndEffectorJacobians,
    EndEffectorHessians,
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
        .value("Action", Action)
        .value("JointAngles", JointAngles)
        .value("JointVelocities", JointVelocities)
        .value("EndEffectorPoints", EndEffectorPoints)
        .value("EndEffectorPointVelocities", EndEffectorPointVelocities)
        .value("EndEffectorPositions", EndEffectorPositions)
        .value("EndEffectorRotations", EndEffectorRotations)
        .value("EndEffectorJacobians", EndEffectorJacobians)
        .value("EndEffectorHessians", EndEffectorJacobians)
        .value("RGBImage",RGBImage)
        .value("TotalDataTypes", TotalDataTypes)
        ;
}

#endif
