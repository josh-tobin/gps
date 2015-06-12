/*
Base class for a controller. Controllers take in sensor readings and choose the action.
*/
#pragma once

// Headers.
#include <boost/scoped_ptr.hpp>

namespace gps_control
{

// Forward declarations.
class sample;

class controller
{
private:

public:
    // Constructor.
    controller();
    // Destructor.
    virtual ~controller();
    // Update the controller (take an action).
    virtual void update(robot_plugin *plugin, double sec_elapsed, std::vector<sensor> &sensors) = 0;
    // Check if controller is finished with its current task.
    virtual bool is_finished() const = 0;
    // Ask the controller to return the sample collected from its latest execution.
    virtual boost::scoped_ptr<sample> get_sample() const = 0;
};

}

/*
TODO: figure out how commands are passed to the controllers.
*/
