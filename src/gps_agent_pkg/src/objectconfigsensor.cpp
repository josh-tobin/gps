#include "gps_agent_pkg/objectconfigsensor.h"
#include "gps_agent_pkg/robotplugin.h"
#include <string>
#include "ros/ros.h"
using namespace gps_control;

// Constructor.
ObjectConfigSensor::ObjectConfigSensor(ros::NodeHandle& n, RobotPlugin *plugin, std::string object_name): Sensor(n, plugin)
{ 
    mass_.resize(1,1);
    mass_.fill(-1); // Mass is -1 if we haven't observed it yet
    object_name_ = object_name; 
    client = n.serviceClient<gazebo_msgs::GetLinkProperties>("/gazebo/get_link_properties"); 
    link_id.request.link_name = object_name_;
}

ObjectConfigSensor::~ObjectConfigSensor()
{
    // Nothing to do here.
}

// Update the sensor (called every tick).
void ObjectConfigSensor::update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step)
{

    //mass_.fill(0.1);   
    //if(is_controller_step) {
    //    mass_.fill(0.1);
    //}
           
    client.call(link_id);
    mass_.fill(link_id.response.mass);
}

void ObjectConfigSensor::configure_sensor(OptionsMap &options)
{
    // Nothing to do here.
}

void ObjectConfigSensor::set_sample_data_format(boost::scoped_ptr<Sample>& sample)
{
    OptionsMap envconf_metadata;
    // Hard code size for now.
    sample->set_meta_data(gps::ENV_CONF, 1, SampleDataFormatEigenVector, envconf_metadata);
}

void ObjectConfigSensor::set_sample_data(boost::scoped_ptr<Sample>& sample, int t)
{
// Set mass.
    // where does this come from? What is SampleDataFormatEig...
    sample->set_data_vector(t, gps::ENV_CONF, mass_.data(), 1, SampleDataFormatEigenVector);
}
