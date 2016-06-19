/*
 * Controller that executes a trial on only a limited number of joints
 * of the PR2. The rest are held in place by a PID controller.
 */
#pragma once

// Headers.
#include <vector>
#include <Eigen/Dense>

//Superclass

#include "gps_agent_pkg/positioncontroller.h"
#include "gps_agent_pkg/tfcontroller.h"

namespace gps_control
{
    class LowDofController : public TfController
    {
    private:
        std::vector<int> dofs_;
        PositionController *pd_controller_;
        Eigen::VectorXd U_dofs_;
        Eigen::VectorXd U_pd_; 

    public: 
        LowDofController(ros::NodeHandle &n, gps::ActuatorType arm, 
                         int size);
        virtual void get_action(int t, const Eigen::VectorXd &X, 
                                const Eigen::VectorXd &obs, Eigen::VectorXd &U);
        virtual void configure_controller(OptionsMap &options);
    };
}
