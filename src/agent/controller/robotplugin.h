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

// Convenience defines.
#define ros_publisher_ptr(X) boost::scoped_ptr<realtime_tools::RealtimePublisher<X> >

namespace GPSControl
{

// Forward declarations.
// Controllers.
class PositionController;
class TrialController;
// Sensors.
class Sensor;
// Custom ROS messages.
class TrialResultMsg;
class PositionCommandMsg;
class TrialCommandMsg;
class RelaxCommandMsg;

class RobotPlugin
{
private:
    // Position controller for passive arm.
    boost::scoped_ptr<PositionController> passive_arm_controller_;
    // Position controller for active arm.
    boost::scoped_ptr<PositionController> active_arm_controller_;
    // Current trial controller (if any).
    boost::scoped_ptr<TrialController> trial_controller_;
    // Sensors.
    std::vector<Sensor> sensors_;
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
    // Publish result of a trial (also indicates completion of position command).
    ros_publisher_ptr(TrialResultMsg) trial_publisher_;
    // Publish state report.
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
    // Initialize all of the sensors.
    virtual void initialize_sensors(ros::NodeHandle& n);
    // Reply with current sensor readings.
    virtual void publish_sensor_readings(/* TODO: implement */);
    // Run a trial.
    virtual void run_trial(/* TODO: receive all of the trial parameters here */);
    // Move the arm.
    virtual void move_arm(/* TODO: receive all of the parameters here, including which arm to move */);
    // Subscriber callbacks.
    // Position command callback.
    virtual void position_subscriber_callback(const PositionCommandMsg::ConstPtr& msg);
    // Trial command callback.
    virtual void trial_subscriber_callback(const TrialCommandMsg::ConstPtr& msg);
    // Relax command callback.
    virtual void trial_subscriber_callback(const RelaxCommandMsg::ConstPtr& msg);
    // Report request callback.
    virtual void report_subscriber_callback(const std_msgs::Empty::ConstPtr& msg);
};

}
