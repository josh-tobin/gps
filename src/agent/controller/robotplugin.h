/*
This is the base class for the robot plugin, which takes care of interfacing
with the robot.
*/
#pragma once

// Forward declarations.
class PositionController;
class Sensor;

class RobotPlugin
{
private:
    // Position controller for passive arm.
    boost::scoped_ptr<PositionController> passive_arm_controller_;
    // Position controller for active arm.
    boost::scoped_ptr<PositionController> active_arm_controller_;
    // Sensors.
    std::vector<Sensor> sensors_;
    /* add various ROS-specific variables here, including publishers, subscribers, etc */
    /* note that creating and destroying publishers and subscribers has non-trivial computational cost, so try to do it all up front */
public:
    // Constructor.
    RobotPlugin();
    // Destructor.
    virtual ~RobotPlugin();
    // Initialize everything.
    virtual void initialize();
    // Initialize all of the ROS subscribers and publishers.
    virtual void initialize_ros();
    // Initialize all of the position controllers.
    virtual void initialize_position_controllers();
    // Initialize all of the sensors.
    virtual void initialize_sensors();
    // Reply with current sensor readings.
    virtual void publish_sensor_readings(/* TODO: implement */);
    // Run a trial.
    virtual void run_trial(/* TODO: receive all of the trial parameters here */);
    // Move the arm.
    virtual void move_arm(/* TODO: receive all of the parameters here, including which arm to move */);
    /*

    add callbacks for all of the ROS messages here
    should have the following message:
    - position control (joint space, task space, etc)
    - run a trial
    - run a trial without actuation (for demo)
    - report current state (sensor array)
    - should have *no* state -- all messages contain all state!

    */
};
