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

class LinearGaussianController : public TrialController
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
    LinearGaussianController(ros::NodeHandle& n);
    // Destructor.
    virtual ~LinearGaussianController();
    // Compute the action at the current time step.
    virtual void get_action(int t, const VectorXd &X, const VectorXd &obs, VectorXd &U);
    // Configure the controller.
    virtual void configure_controller(const OptionsMap &options);
};

}
