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
// JointAngle
// JointVelocity
// EndEffectorPoint
// EndEffectorPointVelocity
// EndEffectorPosition
// EndEffectorRotation
// EndEffectorJacobian

namespace gps_control
{

class EncoderSensor: public Sensor
{
private:
    // Previous joint angles.
    std::vector<double> previous_angles_;
    // Previous joint velocities.
    std::vector<double> previous_velocities_;
    // Temporary storage for joint angles.
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
    // Time from last update when the previous angles were recorded (necessary to compute velocities).
    ros::Time previous_angles_time_;
public:
    // Constructor.
    EncoderSensor(ros::NodeHandle& n, RobotPlugin *plugin);
    // Destructor.
    virtual void ~EncoderSensor();
    // Update the sensor (called every tick).
    virtual void update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step);
    // Configure the sensor (for sensor-specific trial settings).
    // The settings include the configuration for the Kalman filter.
    virtual void configure_sensor(const OptionsMap &options);
    // Set data format and meta data on the provided sample.
    virtual boost::scoped_ptr<Sample> set_sample_data_format(boost::scoped_ptr<Sample> sample) const;
    // Set data on the provided sample.
    virtual boost::scoped_ptr<Sample> set_sample_data(boost::scoped_ptr<Sample> sample) const;
};

}
