/*
This is the base class for the robot plugin, which takes care of interfacing
with the robot.
*/
#pragma once

// Headers.
#include <vector>
#include <boost/scoped_ptr.hpp>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <kdl/chain.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>

// Convenience defines.
#define ros_publisher_ptr(X) boost::scoped_ptr<realtime_tools::RealtimePublisher<X> >

namespace gps_control
{

// List of arm types.
enum ArmType
{
    AuxiliaryArm,
    TrialArm
}

// Forward declarations.
// Controllers.
class PositionController;
class TrialController;
// Sensors.
class Sensor;
// Sample.
class Sample;
// Custom ROS messages.
class SampleResult;
class PositionCommand;
class TrialCommand;
class RelaxCommand;

/* TODO: at some point, need to go through and convert all std::vector<double> to VectorXd! */

class RobotPlugin
{
private:
    // Temporary storage for active arm torques to be applied at each step.
    std::vector<double> active_arm_torques_;
    // Temporary storage for passive arm torques to be applied at each step.
    std::vector<double> passive_arm_torques_;
    // Position controller for passive arm.
    boost::scoped_ptr<PositionController> passive_arm_controller_;
    // Position controller for active arm.
    boost::scoped_ptr<PositionController> active_arm_controller_;
    // Current trial controller (if any).
    boost::scoped_ptr<TrialController> trial_controller_;
    // Sensor data for the current time step.
    boost::scoped_ptr<Sample> current_time_step_sample_;
    // Sensors.
    std::vector<Sensor> sensors_;
    // KDL chains for the end-effectors.
    KDL::Chain passive_arm_fk_chain_, active_arm_fk_chain_;
    // KDL solvers for the end-effectors.
    boost::scoped_ptr<KDL::ChainFkSolverPos> passive_arm_fk_solver_, active_arm_fk_solver_;
    // KDL solvers for end-effector Jacobians.
    boost::scoped_ptr<KDL::ChainJntToJacSolver> passive_arm_jac_solver_, active_arm_jac_solver_;
    // Subscribers.
    // Subscriber for position control commands.
    ros::Subscriber position_subscriber_;
    // Subscriber trial commands.
    ros::Subscriber trial_subscriber_;
    // Subscriber for relax commands.
    ros::Subscriber relax_subscriber_;
    // Subscriber for current state report request.
    ros::Subscriber report_subscriber_;
    // Publishers.
    // Publish result of a trial, completion of position command, or just a report.
    ros_publisher_ptr(TrialResultMsg) report_publisher_;
public:
    // Constructor (this should do nothing).
    RobotPlugin();
    // Destructor.
    virtual ~RobotPlugin();
    // Initialize everything.
    virtual void initialize(ros::NodeHandle& n);
    // Initialize all of the ROS subscribers and publishers.
    virtual void initialize_ros(ros::NodeHandle& n);
    // Initialize all of the position controllers.
    virtual void initialize_position_controllers(ros::NodeHandle& n);
    // Initialize all of the sensors (this also includes FK computation objects).
    virtual void initialize_sensors(ros::NodeHandle& n);
    // Publish the specified sample in a report.
    virtual void publish_report(boost::scoped_ptr<Sample> sample);
    // Run a trial.
    virtual void run_trial(/* TODO: receive all of the trial parameters here */);
    // Move the arm.
    virtual void move_arm(/* TODO: receive all of the parameters here, including which arm to move */);
    // Subscriber callbacks.
    // Position command callback.
    virtual void position_subscriber_callback(const PositionCommand::ConstPtr& msg);
    // Trial command callback.
    virtual void trial_subscriber_callback(const TrialCommand::ConstPtr& msg);
    // Relax command callback.
    virtual void relax_subscriber_callback(const RelaxCommand::ConstPtr& msg);
    // Report request callback.
    virtual void report_subscriber_callback(const std_msgs::Empty::ConstPtr& msg);
    // Update functions.
    // Update the sensors at each time step.
    virtual void update_sensors(ros::Time current_time, bool is_controller_step);
    // Update the controllers at each time step.
    virtual void update_controllers(ros::Time current_time, bool is_controller_step);
    // Accessors.
    // Get current time.
    virtual ros::Time get_current_time() const = 0;
    // Get sensor.
    virtual Sensor *get_sensor(SensorType sensor);
    // Get current encoder readings (robot-dependent).
    virtual void get_joint_encoder_readings(VectorXd &angles, ArmType arm) const = 0;
    // Get forward kinematics solver.
    virtual void get_fk_solver(boost::scoped_ptr<KDL::ChainFkSolverPos> &fk_solver, boost::scoped_ptr<KDL::ChainJntToJacSolver> &jac_solver, ArmType arm);
};

}
