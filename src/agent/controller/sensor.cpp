#include "agent/controller/sensor.h"

using namespace gps_control;

// Factory function.
static sensor sensor::create_sensor(sensor_type type, ros::NodeHandle& n, robot_plugin *plugin)
{
    switch (type)
    {
    case encoder_sensor_type:
        sensor = encoder_sensor(n,plugin);
        break;
    case fk_sensor_type:
        sensor = fk_sensor(n,plugin);
        break;
    case camera_sensor_type:
        sensor = camera_sensor(n,plugin);
        break;
    default:
        ROS_ERROR("Unknown sensor type %i requested from sensor constructor!",type);
    }
}

// Constructor.
sensor::sensor(ros::NodeHandle& n, robot_plugin *plugin)
{
    // Nothing to do.
}

// Destructor.
void sensor::~sensor()
{
    // Nothing to do.
}

// Reset the sensor, clearing any previous state and setting it to the current state.
void sensor::reset(robot_plugin *plugin, ros::Time current_time)
{
    // Nothing to do.
}

// Update the sensor (called every tick).
void sensor::update(robot_plugin *plugin, ros::Time current_time, bool is_controller_step)
{
    // Nothing to do.
}

// Set sensor update delay.
void sensor::set_update(double new_sensor_step_length)
{
    sensor_step_length_ = new_sensor_step_length;
}

// Configure the sensor (for sensor-specific trial settings).
void sensor::configure_sensor(const options_map &options)
{
    // Nothing to do.
}
