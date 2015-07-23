/*
Controller that executes a trial, using a control strategy that is defined in
a subclass.
*/
#pragma once

// Headers.
#include <Eigen/Dense>

// Superclass.
#include "agent/controller/controller.h"

namespace gps_control
{

class TrialController : public Controller
{
private:
    // Current time step.
    int t_;
    // Counter for time step increment.
    int step_counter_;
    // Current time step.
    boost::scoped_ptr<Sample> current_step_;
    // Trajectory sample.
    boost::scoped_ptr<Sample> sample_;
public:
    // Constructor.
    TrialController(ros::NodeHandle& n);
    // Destructor.
    virtual ~TrialController();
    // Compute the action at the current time step.
    virtual void get_action(int t, const VectorXd &X, const VectorXd &obs, VectorXd &U) = 0;
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
