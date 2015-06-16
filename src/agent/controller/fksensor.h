/*
Forward kinematics sensor: returns end-effector point positions and, optionally, their velocities.
*/
#pragma once

// Headers.
#include <Eigen/Dense>

// Superclass.
#include "agent/controller/sensor.h"

/*
TODO: this thing needs access to the filtered joint angles from EncoderSensor...
or does it? should we instead have a Kalman filter on the points themselves?
probably best to see how ddp_controller.cpp does it, and mimic the same behavior for now
*/

// This sensor writes to the following data types:
// EndEffectorPoint
// EndEffectorPointVelocity
// EndEffectorPosition
// EndEffectorRotation
// EndEffectorJacobian

namespace gps_control
{

class FKSensor: public Sensor
{
private:
    // Temporary storage for joint angles (used to obtain joint angles from encoder sensor).
    std::vector<double> temp_joint_angles_;
    // Temporary storage for KDL joint angle array.
    KDL::JntArray temp_joint_array_;
    // Temporary storage for KDL tip pose.
    KDL::Frame temp_tip_pose_;
    // Temporary storage for KDL Jacobian.
    KDL::Jacobian temp_jacobian_;
    // End-effector points in the space of the end-effector.
    MatrixXd end_effector_points_;
    // Previous end-effector points.
    MatrixXd previous_end_effector_points_;
    // Velocities of points.
    MatrixXd previous_end_effector_point_velocities_;
    // Temporary storage.
    MatrixXd temp_end_effector_points_;
    // Previous end-effector position.
    Vector3d previous_position_;
    // Previous end-effector rotation.
    Matrix3d previous_rotation_;
    // Previous end-effector Jacobian.
    MatrixXd previous_jacobian_;
    // Time from last update when the previous end-effector pose was recorded (necessary to compute velocities).
    double previous_pose_time_;
public:
    // Constructor.
    FKSensor(ros::NodeHandle& n, RobotPlugin *plugin);
    // Destructor.
    virtual void ~FKSensor();
    // Reset the sensor, clearing any previous state and setting it to the current state.
    virtual void reset(RobotPlugin *plugin, ros::Time current_time);
    // Update the sensor (called every tick).
    virtual void update(RobotPlugin *plugin, double sec_elapsed, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // This function is used to pass the end-effector points.
    virtual void configure_sensor(const OptionsMap &options);
    // Set data format and meta data on the provided sample.
    virtual boost::scoped_ptr<Sample> set_sample_data_format(boost::scoped_ptr<Sample> sample) const;
    // Set data on the provided sample.
    virtual boost::scoped_ptr<Sample> set_sample_data(boost::scoped_ptr<Sample> sample) const;
};

}
