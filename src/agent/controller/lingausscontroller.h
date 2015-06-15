/*
Controller that executes a trial using a time-varying linear-Gaussian
control law.
*/
#pragma once

// Headers.
#include <Eigen/Dense>

// Superclass.
#include "agent/controller/trialcontroller.h"

namespace gps_control
{

class linear_gaussian_controller : public trial_controller
{
private:
    // Nominal trajectory states.
    MatrixXd &X_;
    // Nominal trajectory actions.
    MatrixXd &U_;
    // Linear feedbacks.
    MatrixXd &K_;
public:
    // Constructor.
    linear_gaussian_controller(ros::NodeHandle& n);
    // Destructor.
    virtual ~linear_gaussian_controller();
    // Compute the action at the current time step.
    virtual void get_action(int t, const VectorXd &X, const VectorXd &obs, VectorXd &U);
    // Configure the controller.
    virtual void configure_controller(const options_map &options);
};

}
