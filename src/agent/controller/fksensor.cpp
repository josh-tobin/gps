#include "agent/controller/fksensor.h"

using namespace gps_control;

// Constructor.
fk_sensor::fk_sensor(ros::NodeHandle& n, robot_plugin *plugin)
{
    ROS_ERROR("Not implemented!");
}

// Destructor.
void fk_sensor::~fk_sensor()
{
    ROS_ERROR("Not implemented!");
}

// Update the sensor (called every tick).
void fk_sensor::update(robot_plugin *plugin, double sec_elapsed, bool is_controller_step)
{
    ROS_ERROR("Not implemented!");
}

// Configure the sensor (for sensor-specific trial settings).
// This function is used to pass the end-effector points.
void fk_sensor::configure_sensor(const options_map &options)
{
    /* TODO: note that this will get called every time there is a report, so
    we should not throw out the previous transform just because we are trying
    to set end-effector points. Instead, just use the stored transform to
    compute what the points should be! This will allow us to query positions
    and velocities each time. */
}

// Populate the array of sensor data size and format based on what the sensor wants.
void fk_sensor::get_data_format(std::vector<int> &data_size, std::vector<sample_data_format> &data_format, std::vector<options_map> &data_meta) const
{
    ROS_ERROR("Not implemented!");
}

// Populate the array of sensor data with whatever data this sensor measures.
void fk_sensor::get_data(std::vector<void*> &data, const std::vector<int> &data_size, const std::vector<sample_data_format> &data_format) const
{
    ROS_ERROR("Not implemented!");
}
