/*
Controller that executes a trial using a neural network loaded with Caffe.
*/
#pragma once

// Headers.
#include <Eigen/Dense>

// Superclass.
#include "agent/controller/trialcontroller.h"

namespace gps_control
{

class CaffeController : public TrialController
{
private:
    // Pointer to Caffe network.
    /* TODO: figure this out */
public:
    // Constructor.
    CaffeController(ros::NodeHandle& n);
    // Destructor.
    virtual ~CaffeController();
    // Compute the action at the current time step.
    virtual void get_action(int t, const VectorXd &X, const VectorXd &obs, VectorXd &U);
    // Configure the controller.
    virtual void configure_controller(const OptionsMap &options);
};

}
