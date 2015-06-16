#include "agent/controller/robotplugin.h"
#include "agent/controller/sensor.h"
#include "agent/controller/controller.h"
#include "agent/controller/positioncontroller.h"

using namespace gps_control;

// Plugin constructor.
RobotPlugin::RobotPlugin()
{
    // Nothing to do here, since all variables are initialized in initialize(...)
}

// Destructor.
void RobotPlugin::~RobotPlugin()
{
    // Nothing to do here, since all instance variables are destructed automatically.
}

// Initialize everything.
void RobotPlugin::initialize(ros::NodeHandle& n)
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
void RobotPlugin::initialize_ros(ros::NodeHandle& n)
{
    // Create subscribers.
    position_subscriber_ = n.subscribe("/gps_controller_position_command", 1, &RobotPlugin::position_subscriber_callback, this);
    trial_subscriber_ = n.subscribe("/gps_controller_trial_command", 1, &RobotPlugin::trial_subscriber_callback, this);
    relax_subscriber_ = n.subscribe("/gps_controller_relax_command", 1, &RobotPlugin::relax_subscriber_callback, this);
    report_subscriber_ = n.subscribe("/gps_controller_report_command", 1, &RobotPlugin::report_subscriber_callback, this);

    // Create publishers.
    report_publisher_.reset(new realtime_tools::RealtimePublisher<report_msg>(n, "/gps_controller_report", 1));
}

// Initialize all sensors.
void RobotPlugin::initialize_sensors(ros::NodeHandle& n)
{
    // Clear out the old sensors.
    sensors_.clear();

    // Create all sensors.
    for (SensorType i = 0; i < SensorType::TotalSensorTypes; i++)
    {
        sensors_.push_back(Sensor::create_sensor(i,n,this));
    }

    // Create current state sample and populate it using the sensors.
    current_time_step_sample_.reset(new Sample(1));
    current_time_step_sample_ = initialize_sample(current_time_step_sample_);
}

// Initialize position controllers.
void RobotPlugin::initialize_position_controllers(n)
{
    // Create passive arm position controller.
    passive_arm_controller_.reset(new PositionController(n,ArmType.PassiveArm));

    // Create active arm position controller.
    active_arm_controller_.reset(new PositionController(n,ArmType.ActiveArm));
}

// Helper function to initialize a sample from the current sensors.
void RobotPlugin::initialize_sample(boost::scoped_ptr<Sample> sample)
{
    // Go through all of the sensors and initialize metadata.
    for (SensorType i = 0; i < SensorType::TotalSensorTypes; i++)
    {
        current_time_step_sample_ = sensors_[i].set_sample_data_format(current_time_step_sample_);
    }
}

// Update the sensors at each time step.
void RobotPlugin::update_sensors(ros::Time current_time, bool is_controller_step)
{
    // Update all of the sensors and fill in the sample.
    for (SensorType sensor = 0; sensor < SensorType.TotalSensorTypes; sensor++)
    {
        sensors_[sensor].update(this,last_update_time_,is_controller_step);
        current_time_step_sample_ = sensors_[i].set_sample_data(current_time_step_sample_);
    }
}

// Update the controllers at each time step.
void RobotPlugin::update_controllers(ros::Time current_time, bool is_controller_step)
{
    // If we have a trial controller, update that, otherwise update position controller.
    if (trial_controller_ != NULL) trial_controller_->update(this,last_update_time_,is_controller_step,active_arm_torques_);
    else active_arm_controller_->update(this,last_update_time_,is_controller_step,active_arm_torques_);

    // Update passive arm controller.
    passive_arm_controller_->update(this,last_update_time_,is_controller_step,passive_arm_torques_);

    // Check if the trial controller finished and delete it.
    if (trial_controller_->is_finished())
    {
        // Clear the trial controller.
        trial_controller_.reset(NULL);

        // Switch the sensors to run at full frequency.
        for (SensorType sensor = 0; sensor < SensorType.TotalSensorTypes; sensor++)
        {
            sensors_[sensor].set_update(active_arm_controller_->get_update_delay());
        }
    }
}

// Get sensor.
Sensor *RobotPlugin::get_sensor(SensorType sensor)
{
    assert(sensor < SensorType.TotalSensors);
    return &sensors_[sensor];
}

// Get forward kinematics solver.
void RobotPlugin::get_fk_solver(boost::scoped_ptr<KDL::ChainFkSolverPos> &fk_solver, boost::scoped_ptr<KDL::ChainJntToJacSolver> &jac_solver, ArmType arm)
{
    if (arm == ArmType.PassiveArm)
    {
        fk_solver = passive_arm_fk_solver_;
        jac_solver = passive_arm_jac_solver_;
    }
    else if (arm == ArmType.ActiveArm)
    {
        fk_solver = active_arm_fk_solver_;
        jac_solver = active_arm_jac_solver_;
    }
    else
    {
        ROS_ERROR("Unknown ArmType %i requested for joint encoder readings!",arm);
    }
}
