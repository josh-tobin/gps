/*
The options object is used to list a map from strings to parameters.
*/
#pragma once

// Headers.
#include <map>
#include <string>
#include <boost/variant.hpp>
#include <Eigen/Dense>

namespace gps_control
{

// Types of data supported for internal data storage.
enum options_data_format
{
    data_format_bool,
    data_format_uint8,
    data_format_int,
    data_format_double,
    data_format_matrix,
    data_format_string
};

// This is a parameter entry. Note that the arguments should match the enum.
typedef options_variant boost::variant<bool,uint8_t,int,double,MatrixXd,std::string>;

// This is the options map.
typedef options_map std::map<std::string,options_variant>;

}

/* TODO: provide a utility function that makes it easy to convert one of these maps to a ROS message */
/* TODO: provide a utility function that makes it easy to convert a Python dictionary to one of these */
/* TODO: in the future, might want to move it to some other directory where it will be shared with other C++ components */
