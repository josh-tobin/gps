#include "agent/controller/robotplugin.h"
#include "agent/controller/sensor.h"
#include "agent/controller/controller.h"
#include "agent/controller/positioncontroller.h"

using namespace gps_control;

// Plugin constructor.
robot_plugin::robot_plugin()
{
    // Nothing to do here, since all variables are initialized in initialize(...)
}

// Destructor.
void robot_plugin::~robot_plugin()
{
    // Nothing to do here, since all instance variables are destructed automatically.
}

// Initialize everything.
void robot_plugin::initialize(ros::NodeHandle& n)
{
    // Initialize all ROS communication infrastructure.
    initialize_ros(n);

    // Initialize all sensors.
    initialize_sensors(n);

    // Initialize the position controllers.
    // Note that the trial controllers are created from scratch for each trial.
    // However, the position controllers persist, since there is only one type.
    initialize_position_controllers(n);

    // After this, we still need to create the kinematics solvers. How these are
    // created depends on the particular robot, and should be implemented in a
    // subclass.
}

// Initialize ROS communication infrastructure.
void robot_plugin::initialize_ros(ros::NodeHandle& n)
{
    // Create subscribers.
    position_subscriber_ = n.subscribe("/gps_controller_position_command", 1, &robot_plugin::position_subscriber_callback, this);
    trial_subscriber_ = n.subscribe("/gps_controller_trial_command", 1, &robot_plugin::trial_subscriber_callback, this);
    relax_subscriber_ = n.subscribe("/gps_controller_relax_command", 1, &robot_plugin::relax_subscriber_callback, this);
    report_subscriber_ = n.subscribe("/gps_controller_report_command", 1, &robot_plugin::report_subscriber_callback, this);

    // Create publishers.
    report_publisher_.reset(new realtime_tools::RealtimePublisher<report_msg>(n, "/gps_controller_report", 1));
}

// Initialize all sensors.
void robot_plugin::initialize_sensors(ros::NodeHandle& n)
{
    // Clear out the old sensors.
    sensors_.clear();

    // Create all sensors.
    for (sensor_type i = 0; i < sensor_type::total_sensor_types; i++)
    {
        sensors_.push_back(sensor::create_sensor(i,n,this));
    }
}

// Initialize position controllers.
void robot_plugin::initialize_position_controllers(n)
{
    // Create passive arm position controller.
    passive_arm_controller_.reset(new position_controller(n));

    // Create active arm position controller.
    active_arm_controller_.reset(new position_controller(n));
}
