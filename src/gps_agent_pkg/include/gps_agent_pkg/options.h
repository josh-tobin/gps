/*
The options object is used to list a map from strings to parameters.
*/
#pragma once

// Headers.
#include <map>
#include <vector>
#include <string>
#include <boost/variant.hpp>
#include <Eigen/Dense>

namespace gps_control
{

/* TODO: can we do this with protobuffers instead? */
// Types of data supported for internal data storage.
enum OptionsDataFormat
{
    OptionsDataFormatBool,
    OptionsDataFormatUInt8,
    OptionsDataFormatIntVector,
    OptionsDataFormatInt,
    OptionsDataFormatDouble,
    OptionsDataFormatMatrix,
    OptionsDataFormatVector,
    OptionsDataFormatString
};

// This is a parameter entry. Note that the arguments should match the enum.
typedef boost::variant<bool,uint8_t,std::vector<int>,int,double,Eigen::MatrixXd,Eigen::VectorXd,std::string> OptionsVariant;

// This is the options map.
typedef std::map<std::string,OptionsVariant> OptionsMap;

}

/* TODO: provide a utility function that makes it easy to convert one of these maps to a ROS message */
/* TODO: provide a utility function that makes it easy to convert a Python dictionary to one of these */
/* TODO: in the future, might want to move it to some other directory where it will be shared with other C++ components */
