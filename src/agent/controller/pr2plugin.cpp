#include "agent/controller/pr2plugin.h"

using namespace gps_control;

// Plugin constructor.
pr2_plugin::pr2_plugin()
{
    // Nothing to do here, since all variables are initialized in initialize(...)
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

    // Initialize ROS subscribers/publishers, sensors, and position controllers.
    initialize(n);

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

    // Tell the PR2 controller manager that we initialized everything successfully.
    return true;
}

// 

