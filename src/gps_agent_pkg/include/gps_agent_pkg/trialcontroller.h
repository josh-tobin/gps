/*
Controller that executes a trial, using a control strategy that is defined in
a subclass.
*/
#pragma once

// Headers.
#include <Eigen/Dense>
#include <boost/scoped_ptr.hpp>

#include "gps_agent_pkg/ArmType.h"

// Superclass.
#include "gps_agent_pkg/controller.h"

namespace gps_control
{

class TrialController : public Controller
{
private:
    // Current time step.
    int t_;
    ros::Time last_update_time_;
    // Counter for time step increment.
    int step_counter_;
    // Current time step.
    boost::scoped_ptr<Sample> current_step_;
    // Trajectory sample.
    boost::scoped_ptr<Sample> sample_;
public:
    // Constructor.
    TrialController();
    // Destructor.
    virtual ~TrialController();
    // Compute the action at the current time step.
    virtual void get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U) = 0;
    // Update the controller (take an action).
    virtual void update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques);
    // Configure the controller.
    virtual void configure_controller(OptionsMap &options);
    // Check if controller is finished with its current task.
    virtual bool is_finished() const;
    // Ask the controller to return the sample collected from its latest execution.
    virtual boost::scoped_ptr<Sample>* get_sample() const;
    // Called when controller is turned on
    virtual void reset(ros::Time update_time);
};

}
