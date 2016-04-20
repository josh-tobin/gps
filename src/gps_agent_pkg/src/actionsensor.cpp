#include "gps_agent_pkg/actionsensor.h"
#include "gps_agent_pkg/robotplugin.h"

using namespace gps_control;

// Constructor
ActionSensor::ActionSensor(ros::NodeHandle& n, RobotPlugin *plugin,
                           gps::ActuatorType actuator_type): Sensor(n, plugin)
{
    //previous_action_.resize(7,1);
    //previous_action_.fill(0.);
}

ActionSensor::~ActionSensor()
{
    // Nothing to do here
}

// Update the sensor
void ActionSensor::update(RobotPlugin *plugin, ros::Time current_time,
                                bool is_controller_step)
{
    if (is_controller_step) {
        //previous_action_ = plugin->latest_action_command();
        previous_action_ = plugin->active_arm_torques();
    }
    //previous_action_.fill(0.1);
}

void ActionSensor::configure_sensor(OptionsMap &options)
{
    // Nothing to do here
}

void ActionSensor::set_sample_data_format(boost::scoped_ptr<Sample>& sample)
{
    OptionsMap action_metadata;
    sample->set_meta_data(gps::ACTION, previous_action_.size(), 
                          SampleDataFormatEigenVector, action_metadata);
}

void ActionSensor::set_sample_data(boost::scoped_ptr<Sample>& sample,
                                   int t)
{
    sample->set_data_vector(t, gps::ACTION, previous_action_.data(),
                            previous_action_.size(), 
                            SampleDataFormatEigenVector);
}

