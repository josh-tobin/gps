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
    // Current target (joint space).
    VectorXd target_angles_;
    // Current target (task space).
    MatrixXd target_pose_;
    // Current mode.
    position_control_mode mode_;
    // Time since motion start.
    double start_time_;
public:
    // Constructor.
    PositionController(ros::NodeHandle& n, ArmType arm);
    // Destructor.
    virtual ~PositionController();
    // Update the controller (take an action).
    virtual void update(RobotPlugin *plugin, double sec_elapsed, std::scopted_ptr<Sample> sample);
    // Configure the controller.
    virtual void configure_controller(const OptionsMap &options);
    // Check if controller is finished with its current task.
    virtual bool is_finished() const;
    // Ask the controller to return the sample collected from its latest execution.
    virtual boost::scoped_ptr<Sample> get_sample() const;
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
