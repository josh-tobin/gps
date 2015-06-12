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

class camera_sensor: public sensor
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
    camera_sensor(ros::NodeHandle& n);
    // Destructor.
    virtual ~camera_sensor();
    // Update the sensor (called every tick).
    virtual void update(robot_plugin *plugin, double sec_elapsed, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // This function is used to set resolution, cropping, topic to listen to...
    virtual void configure_sensor(/* TODO: figure out the format of the configuration... some map from strings to options?? */);
};

}
