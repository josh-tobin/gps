/*
Joint encoder sensor: returns joint angles and, optionally, their velocities.
*/
#pragma once

// Superclass.
#include "agent/controller/sensor.h"

/*
TODO: this thing needs a Kalman filter.
*/

// This sensor writes to the following data types:
// JointAngles
// JointVelocities

namespace gps_control
{

class encoder_sensor: public sensor
{
private:
    // Previous joint angles.
    std::vector<double> previous_angles_;
    // Time from last update when the previous angles were recorded (necessary to compute velocities).
    double previous_angles_time_;
public:
    // Constructor.
    encoder_sensor(ros::NodeHandle& n, robot_plugin *plugin);
    // Destructor.
    virtual ~encoder_sensor();
    // Update the sensor (called every tick).
    virtual void update(robot_plugin *plugin, double sec_elapsed, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // The settings include the configuration for the Kalman filter.
    virtual void configure_sensor(/* TODO: figure out the format of the configuration... some map from strings to options?? */);
    // Populate the array of sensor data size and format based on what the sensor wants.
    virtual void get_data_format(std::vector<int> &data_size, std::vector<sample_data_format> &data_format, std::vector<sample_data_meta> &data_meta) const;
    // Populate the array of sensor data with whatever data this sensor measures.
    virtual void get_data(std::vector<void*> &data, const std::vector<int> &data_size, const std::vector<sample_data_format> &data_format) const;
};

}
