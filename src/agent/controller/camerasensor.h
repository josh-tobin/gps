/*
Camera sensor: records latest images from camera.
*/
#pragma once

// Superclass.
#include "agent/controller/sensor.h"

// This sensor writes to the following data types:
// RGBImage

namespace gps_control
{

class CameraSensor: public Sensor
{
private:
    // Latest image.
    std::vector<uint8_t> latest_image_;
    // Time at which image was received.
    double latest_image_time_;
    /* TODO: add variables for current configuration: cropping, resolution, topic name, etc */
    /* TODO: add ROS variables, subscriber, etc */
public:
    // Constructor.
    CameraSensor(ros::NodeHandle& n, RobotPlugin *plugin);
    // Destructor.
    virtual ~CameraSensor();
    // Update the sensor (called every tick).
    virtual void update(RobotPlugin *plugin, double sec_elapsed, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // This function is used to set resolution, cropping, topic to listen to...
    virtual void configure_sensor(const OptionsMap &options);
    // Set data format and meta data on the provided sample.
    virtual boost::scoped_ptr<Sample> set_sample_data_format(boost::scoped_ptr<Sample> sample) const;
    // Set data on the provided sample.
    virtual boost::scoped_ptr<Sample> set_sample_data(boost::scoped_ptr<Sample> sample) const;
};

}
