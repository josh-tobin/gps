/*
 * Controller that executes an open loop policy
 */
#pragma once

// Headers
#include <vector>
#include <Eigen/Dense>

// Superclass
#include "gps_agent_pkg/trialcontroller.h"

namespace gps_control
{

class OpenLoopController : public TrialController
{
private:
    // Control sequence
    std::vector<Eigen::VectorXd> U_;

public:
    // Constructor
    OpenLoopController();
    // Destructor
    virtual ~OpenLoopController();
    // Compute the action at the current time step
    virtual void get_action(int t, const Eigen::VectorXd &X, 
                            const Eigen::VectorXd &obs, Eigen::VectorXd &U);
    // Configure the controller
    virtual void configure_controller(OptionsMap &options);
};

}
