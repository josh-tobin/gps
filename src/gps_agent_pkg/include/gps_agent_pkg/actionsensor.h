/*
 * Action sensor: returns actions taken at previous timestep
 */
#pragma once

#include <boost/shared_ptr.hpp>
#include <Eigen/Dense>
#include "gps/proto/gps.pb.h"

//Superclass
#include "gps_agent_pkg/sensor.h"
#include "gps_agent_pkg/sample.h"

// This sensor writes to the following data types:
// Action
//
namespace gps_control
{

class ActionSensor: public Sensor
{
private:
    // Previous action.
    Eigen::VectorXd previous_action_;
public:
    ActionSensor(ros::NodeHandle& n, RobotPlugin *plugin,
                 gps::ActuatorType actuator_type);
    virtual ~ActionSensor();
    virtual void update(RobotPlugin *plugin, ros::Time current_time,
                        bool is_controller_step);
    virtual void configure_sensor(OptionsMap &options);
    virtual void set_sample_data_format(boost::scoped_ptr<Sample>& sample);
    virtual void set_sample_data(boost::scoped_ptr<Sample>& sample, int t);
};

}
