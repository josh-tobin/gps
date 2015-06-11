/*
Camera sensor: records latest images from camera.
*/
#pragma once

// Superclass.
#include "agent/controller/sensor.h"

namespace GPSControl
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
    CameraSensor(ros::NodeHandle& n);
    // Destructor.
    virtual ~CameraSensor();
    // Update the sensor (called every tick).
    virtual void update(RobotPlugin *plugin, double sec_elapsed, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // This function is used to set resolution, cropping, topic to listen to...
    virtual void configure_sensor(/* TODO: figure out the format of the configuration... some map from strings to options?? */);
};

}
