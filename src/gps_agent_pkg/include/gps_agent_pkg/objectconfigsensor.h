/*
Object config sensor: returns the exact config of objects in the scene the pr2
is interacting with.
*/
#pragma once
#include <boost/shared_ptr.hpp>
#include "gps/proto/gps.pb.h"
#include <string>
// Superclass.
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/sample.h"
#include "gps_agent_pkg/encoderfilter.h"
#include <Eigen/Dense>
#include <ros/ros.h>
#include <gazebo_msgs/GetLinkProperties.h>
// This sensor writes to the following data types:
// object mass
// in the future we will have a more flexible way of dealing with object params

namespace gps_control
{

class ObjectConfigSensor: public Sensor
{
private:
    //double *mass_;
    Eigen::VectorXd mass_;
    std::string object_name_;
    //ros::NodeHandle n;
    //ros::ServiceClient client;
   // gazebo_msgs::GetLinkProperties link_id;
public:
    // Constructor
    ObjectConfigSensor(ros::NodeHandle& n, RobotPlugin *plugin, std::string object_name);
    // Destructor 
    virtual ~ObjectConfigSensor();
    // Update the sensor (called every tick).
    virtual void update(ros::Time current_time, bool is_controller_step);
    // Configure the sensor (needed?)
    virtual void configure_sensor(OptionsMap &options);
    // Set data format and meta data on the provided sample.
    virtual void set_sample_data_format(boost::scoped_ptr<Sample>& sample);
    void set_sample_data(boost::scoped_ptr<Sample>& sample, int t);
};

}
