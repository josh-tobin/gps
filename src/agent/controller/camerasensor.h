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
    camera_sensor(ros::NodeHandle& n, robot_plugin *plugin);
    // Destructor.
    virtual ~camera_sensor();
    // Update the sensor (called every tick).
    virtual void update(robot_plugin *plugin, double sec_elapsed, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // This function is used to set resolution, cropping, topic to listen to...
    virtual void configure_sensor(/* TODO: figure out the format of the configuration... some map from strings to options?? */);
    // Populate the array of sensor data size and format based on what the sensor wants.
    virtual void get_data_format(std::vector<int> &data_size, std::vector<sample_data_format> &data_format, std::vector<sample_data_meta> &data_meta) const;
    // Populate the array of sensor data with whatever data this sensor measures.
    virtual void get_data(std::vector<void*> &data, const std::vector<int> &data_size, const std::vector<sample_data_format> &data_format) const;
};

}
