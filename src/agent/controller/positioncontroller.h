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
enum position_control_mode
{
    no_control,
    joint_space_control,
    task_space_control
};

class position_controller : public controller
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
    position_controller();
    // Destructor.
    virtual ~position_controller();
    // Update the controller (take an action).
    virtual void update(robot_plugin *plugin, double sec_elapsed, std::vector<sensor> &sensors);
    // Check if controller is finished with its current task.
    virtual bool is_finished();
    // Ask the controller to return the sample collected from its latest execution.
    virtual boost::scoped_ptr<sample> get_sample();
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
