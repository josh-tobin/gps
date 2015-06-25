/*
Controller that moves the arm to a position, either in joint space or in task
space.
*/
#pragma once

// Headers.
#include <Eigen/Dense>

// Superclass.
#include "agent/controller/controller.h"

namespace gps_control
{

// Current motion type.
enum PositionControlMode
{
    NoControl,
    JointSpaceControl,
    TaskSpaceControl
};

class PositionController : public Controller
{
private:
    // P gains.
    VectorXd pd_gains_p_;
    // D gains.
    VectorXd pd_gains_d_;
    // I gains.
    VectorXd pd_gains_i_;
    // Integral terms.
    VectorXd pd_integral_;
    // Maximum joint velocities.
    VectorXd max_velocities_;
    // Temporary storage for Jacobian.
    MatrixXd temp_jacobian_;
    // Temporary storage for joint angle offset.
    VectorXd temp_angles_;
    // Current target (joint space).
    VectorXd target_angles_;
    // Current target (task space).
    VectorXd target_pose_;
    // Latest joint angles.
    VectorXd current_angles_;
    // Latest joint angle velocities.
    VectorXd current_angle_velocities_;
    // Latest pose.
    VectorXd current_pose_;
    // Current mode.
    PositionControlMode mode_;
    // Current arm.
    ArmType arm_;
    // Time since motion start.
    ros::Time start_time_;
    // Time of last update.
    ros::Time last_update_time_;
public:
    // Constructor.
    PositionController(ros::NodeHandle& n, ArmType arm);
    // Destructor.
    virtual void ~PositionController();
    // Update the controller (take an action).
    virtual void update(RobotPlugin *plugin, ros::Time current_time, std::scopted_ptr<Sample> sample, std::vector<double> &torques);
    // Configure the controller.
    virtual void configure_controller(const OptionsMap &options);
    // Check if controller is finished with its current task.
    virtual bool is_finished() const;
    // Ask the controller to return the sample collected from its latest execution.
    virtual boost::scoped_ptr<Sample> get_sample() const;
    // Reset the controller -- this is typically called when the controller is turned on.
    virtual void reset();
};

}

/*

note that this thing may be running at the same time as the trial controller in order to control the left arm

would be good to have the following functionality:

1. go to a position (with maximum speed limit and high gains)

2. go to a position task space (Via J^T control)

3. hold position with variable stiffness

4. signal when at position with a timeout and variable stiffness (integral term?)

*/
