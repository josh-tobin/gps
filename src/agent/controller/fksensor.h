/*
Forward kinematics sensor: returns end-effector point positions and, optionally, their velocities.
*/
#pragma once

// Headers.
#include <Eigen/Dense>

// Superclass.
#include "agent/controller/sensor.h"

/*
TODO: this thing needs access to the filtered joint angles from encodersensor...
or does it? should we instead have a Kalman filter on the points themselves?
probably best to see how ddp_controller.cpp does it, and mimic the same behavior for now
*/

namespace GPSControl
{

class FKSensor: public Sensor
{
private:
    // End-effector points in the space of the end-effector.
    MatrixXd end_effector_points_;
    // Previous end-effector transform.
    Matrix4d previous_transform_;
    // Time from last update when the previous end-effector pose was recorded (necessary to compute velocities).
    double previous_pose_time_;
public:
    // Constructor.
    FKSensor(ros::NodeHandle& n);
    // Destructor.
    virtual ~FKSensor();
    // Update the sensor (called every tick).
    virtual void update(RobotPlugin *plugin, double sec_elapsed, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // This function is used to pass the end-effector points.
    virtual void configure_sensor(/* TODO: figure out the format of the configuration... some map from strings to options?? */);
};

}
