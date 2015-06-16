/*
Base class for a controller. Controllers take in sensor readings and choose the action.
*/
#pragma once

// Headers.
#include <boost/scoped_ptr.hpp>

// This allows us to use options.
#include "options.h"

namespace gps_control
{

// Forward declarations.
class Sample;

class Controller
{
private:

public:
    // Constructor.
    Controller(ros::NodeHandle& n, ArmType arm);
    // Destructor.
    virtual ~Controller();
    // Update the controller (take an action).
    virtual void update(RobotPlugin *plugin, double sec_elapsed, std::scopted_ptr<Sample> sample) = 0;
    // Configure the controller.
    virtual void configure_controller(const OptionsMap &options);
    // Set update delay on the controller.
    virtual void set_update_delay(double new_step_length);
    // Get update delay on the controller.
    virtual double get_update_delay();
    // Check if controller is finished with its current task.
    virtual bool is_finished() const = 0;
    // Ask the controller to return the sample collected from its latest execution.
    virtual boost::scoped_ptr<Sample> get_sample() const = 0;
};

}

/*
TODO: figure out how commands are passed to the controllers.
*/
