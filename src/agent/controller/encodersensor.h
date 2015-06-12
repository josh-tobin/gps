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
    encoder_sensor(ros::NodeHandle& n);
    // Destructor.
    virtual ~encoder_sensor();
    // Update the sensor (called every tick).
    virtual void update(robot_plugin *plugin, double sec_elapsed, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // The settings include the configuration for the Kalman filter.
    virtual void configure_sensor(/* TODO: figure out the format of the configuration... some map from strings to options?? */);
};

}
