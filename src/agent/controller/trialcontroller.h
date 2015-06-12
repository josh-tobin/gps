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

class position_controller : public controller
{
private:
    // Current time step.
    int t_;
    // Counter for time step increment.
    int step_counter_;
    // Current sample.
    boost::scoped_ptr<sample> sample_;
public:
    // Constructor.
    position_controller();
    // Destructor.
    virtual ~position_controller();
    // Compute the action at the current time step.
    virtual void get_action(int t, const VectorXd &X, const VectorXd &obs, VectorXd &U) = 0;
    // Update the controller (take an action).
    virtual void update(robot_plugin *plugin, double sec_elapsed, std::scopted_ptr<sample> sample);
    // Configure the controller.
    virtual void configure_controller(/* TODO: figure out the format of the configuration... some map from strings to options?? */);
    // Check if controller is finished with its current task.
    virtual bool is_finished() const;
    // Ask the controller to return the sample collected from its latest execution.
    virtual boost::scoped_ptr<sample> get_sample() const;
};

}
