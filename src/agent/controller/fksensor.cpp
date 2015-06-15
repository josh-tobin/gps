#include "agent/controller/fksensor.h"

using namespace gps_control;

// Constructor.
FKSensor::FKSensor(ros::NodeHandle& n, RobotPlugin *plugin)
{
    ROS_ERROR("Not implemented!");
}

// Destructor.
void FKSensor::~FKSensor()
{
    ROS_ERROR("Not implemented!");
}

// Update the sensor (called every tick).
void FKSensor::update(RobotPlugin *plugin, double sec_elapsed, bool is_controller_step)
{
    ROS_ERROR("Not implemented!");
}

// Configure the sensor (for sensor-specific trial settings).
// This function is used to pass the end-effector points.
void FKSensor::configure_sensor(const OptionsMap &options)
{
    /* TODO: note that this will get called every time there is a report, so
    we should not throw out the previous transform just because we are trying
    to set end-effector points. Instead, just use the stored transform to
    compute what the points should be! This will allow us to query positions
    and velocities each time. */
}

// Populate the array of sensor data size and format based on what the sensor wants.
void FKSensor::get_data_format(std::vector<int> &data_size, std::vector<SampleDataFormat> &data_format, std::vector<OptionsMap> &data_meta) const
{
    ROS_ERROR("Not implemented!");
}

// Populate the array of sensor data with whatever data this sensor measures.
void FKSensor::get_data(std::vector<void*> &data, const std::vector<int> &data_size, const std::vector<SampleDataFormat> &data_format) const
{
    ROS_ERROR("Not implemented!");
}
