/*
Forward kinematics sensor: returns end-effector point positions and, optionally, their velocities.
*/
#pragma once

// Headers.
#include <Eigen/Dense>

// Superclass.
#include "agent/controller/sensor.h"

/*
TODO: this thing needs access to the filtered joint angles from encoder_sensor...
or does it? should we instead have a Kalman filter on the points themselves?
probably best to see how ddp_controller.cpp does it, and mimic the same behavior for now
*/

// This sensor writes to the following data types:
// EndEffectorPoints
// EndEffectorVelocities
// EndEffectorPosition
// EndEffectorRotation
// EndEffectorJacobian

namespace gps_control
{

class fk_sensor: public sensor
{
private:
    // End-effector points in the space of the end-effector.
    MatrixXd end_effector_points_;
    // Previous end-effector transform.
    Matrix4d previous_transform_;
    // Previous end-effector Jacobian.
    MatrixXd previous_jacobian_;
    // Time from last update when the previous end-effector pose was recorded (necessary to compute velocities).
    double previous_pose_time_;
public:
    // Constructor.
    fk_sensor(ros::NodeHandle& n, robot_plugin *plugin);
    // Destructor.
    virtual void ~fk_sensor();
    // Reset the sensor, clearing any previous state and setting it to the current state.
    virtual void reset(robot_plugin *plugin, ros::Time current_time);
    // Update the sensor (called every tick).
    virtual void update(robot_plugin *plugin, double sec_elapsed, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // This function is used to pass the end-effector points.
    virtual void configure_sensor(const options_map &options);
    // Populate the array of sensor data size and format based on what the sensor wants.
    virtual void get_data_format(std::vector<int> &data_size, std::vector<sample_data_format> &data_format, std::vector<options_map> &data_meta) const;
    // Populate the array of sensor data with whatever data this sensor measures.
    virtual void get_data(std::vector<void*> &data, const std::vector<int> &data_size, const std::vector<sample_data_format> &data_format) const;
};

}
