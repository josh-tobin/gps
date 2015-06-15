#include "agent/controller/sensor.h"

using namespace gps_control;

// Factory function.
static Sensor Sensor::create_sensor(SensorType type, ros::NodeHandle& n, RobotPlugin *plugin)
{
    switch (type)
    {
    case EncoderSensorType:
        return EncoderSensor(n,plugin);
    case FKSensorType:
        return FKSensor(n,plugin);
    case CameraSensorType:
        return CameraSensor(n,plugin);
    default:
        ROS_ERROR("Unknown sensor type %i requested from sensor constructor!",type);
    }
}

// Constructor.
Sensor::Sensor(ros::NodeHandle& n, RobotPlugin *plugin)
{
    // Nothing to do.
}

// Destructor.
void Sensor::~Sensor()
{
    // Nothing to do.
}

// Reset the sensor, clearing any previous state and setting it to the current state.
void Sensor::reset(RobotPlugin *plugin, ros::Time current_time)
{
    // Nothing to do.
}

// Update the sensor (called every tick).
void Sensor::update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step)
{
    // Nothing to do.
}

// Set sensor update delay.
void Sensor::set_update(double new_sensor_step_length)
{
    sensor_step_length_ = new_sensor_step_length;
}

// Configure the sensor (for sensor-specific trial settings).
void Sensor::configure_sensor(const OptionsMap &options)
{
    // Nothing to do.
}
