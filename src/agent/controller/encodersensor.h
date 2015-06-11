/*
Joint encoder sensor: returns joint angles and, optionally, their velocities.
*/
#pragma once

// Superclass.
#include "agent/controller/sensor.h"

/*
TODO: this thing needs a Kalman filter.
*/

namespace GPSControl
{

class EncoderSensor: public Sensor
{
private:
    // Previous joint angles.
    std::vector<double> previous_angles_;
    // Time from last update when the previous angles were recorded (necessary to compute velocities).
    double previous_angles_time_;
public:
    // Constructor.
    EncoderSensor(ros::NodeHandle& n);
    // Destructor.
    virtual ~EncoderSensor();
    // Update the sensor (called every tick).
    virtual void update(RobotPlugin *plugin, double sec_elapsed, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // The settings include the configuration for the Kalman filter.
    virtual void configure_sensor(/* TODO: figure out the format of the configuration... some map from strings to options?? */);
};

}
