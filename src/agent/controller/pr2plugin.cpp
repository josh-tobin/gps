#include "agent/controller/pr2plugin.h"

using namespace gps_control;

// Plugin constructor.
pr2_plugin::pr2_plugin()
{
    // Some basic variable initialization.
    controller_counter_ = 0;
    controller_step_length_ = 50;
}

// Destructor.
void pr2_plugin::~pr2_plugin()
{
    // Nothing to do here, since all instance variables are destructed automatically.
}

// Initialize the object and store the robot state.
bool pr2_plugin::init(pr2_mechanism_model::RobotState* robot, ros::NodeHandle& n)
{
    // Variables.
    std::string root_name, active_tip_name, passive_tip_name;

    // Store the robot state.
    robot_ = robot;

    // Create FK solvers.
    // Get the name of the root.
    if(!n.getParam("root_name", root_name)) {
        ROS_ERROR("Property root_name not found in namespace: '%s'", n.getNamespace().c_str());
        return false;
    }

    // Get active and passive arm end-effector names.
    if(!n.getParam("active_tip_name", active_tip_name)) {
        ROS_ERROR("Property active_tip_name not found in namespace: '%s'", n.getNamespace().c_str());
        return false;
    }
    if(!n.getParam("passive_tip_name", passive_tip_name)) {
        ROS_ERROR("Property passive_tip_name not found in namespace: '%s'", n.getNamespace().c_str());
        return false;
    }

    // Create active arm chain.
    if(!active_arm_chain_.init(robot_, root_name, active_tip_name)) {
        ROS_ERROR("Controller could not use the chain from '%s' to '%s'", root_name.c_str(), active_tip_name.c_str());
        return false;
    }

    // Create passive arm chain.
    if(!passive_arm_chain_.init(robot_, root_name, passive_tip_name)) {
        ROS_ERROR("Controller could not use the chain from '%s' to '%s'", root_name.c_str(), passive_tip_name.c_str());
        return false;
    }

    // Create KDL chains, solvers, etc.
    // KDL chains.
    passive_arm_chain_.toKDL(passive_arm_fk_chain_);
    active_arm_chain_.toKDL(active_arm_fk_chain_);

    // Pose solvers.
    passive_arm_fk_solver_.reset(new KDL::ChainFkSolverPos_recursive(passive_arm_fk_chain_));
    active_arm_fk_solver_.reset(new KDL::ChainFkSolverPos_recursive(active_arm_fk_chain_));

    // Jacobian sovlers.
    passive_arm_jac_solver_.reset(new KDL::ChainJntToJacSolver(passive_arm_fk_chain_));
    active_arm_jac_solver_.reset(new KDL::ChainJntToJacSolver(active_arm_fk_chain_));

    // Initialize ROS subscribers/publishers, sensors, and position controllers.
    // Note that this must be done after the FK solvers are created, because the sensors
    // will ask to use these FK solvers!
    initialize(n);

    // Tell the PR2 controller manager that we initialized everything successfully.
    return true;
}

// This is called by the controller manager before starting the controller.
void pr2_plugin::starting()
{
    // Get current time.
    last_update_time_ = robot_->getTime();
    controller_counter_ = 0;

    // Reset all the sensors. This is important for sensors that try to keep
    // track of the previous state somehow.
    for (sensor_type sensor = 0; sensor < sensor_type.total_sensor_types; sensor++)
    {
        sensors_[sensor].reset(this,last_update_time_);
    }

    // Reset position controllers.
    passive_arm_controller_->reset(last_update_time_);
    active_arm_controller_->reset(last_update_time_);

    // Reset trial controller, if any.
    if (trial_controller_ != NULL) trial_controller_->reset(this,last_update_time_);
}

// This is called by the controller manager before stopping the controller.
void pr2_plugin::stopping()
{
    // Nothing to do here.
}

// This is the main update function called by the realtime thread when the controller is running.
void pr2_plugin::update()
{
    // Get current time.
    last_update_time_ = robot_->getTime();

    // Check if this is a controller step based on the current controller frequency.
    controller_counter_++;
    if (controller_counter_ >= controller_step_length_) controller_counter = 0;
    bool is_controller_step = (controller_counter == 0);

    // Update the sensors and fill in the current step sample.
    update_sensors(last_update_time_,is_controller_step);

    // Update the controllers.
    update_controllers(last_update_time_,is_controller_step);
}

// Get current encoder readings (robot-dependent).
void pr2_plugin::get_joint_encoder_readings(std::vector<double> &angles) const
{
    // TODO: check that the angles vector is the same length as the vector in the robot object.


    // TODO: copy over the joint angles.
}
