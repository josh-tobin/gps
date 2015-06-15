/*
The state object maintains the state, assembles state and observation vectors,
and keeps track of what is and is not included in the state. This object is
used both by the controller, to incrementally assemble the state during a
trial, to keep track of sample data and get the state and observation vectors
from it.
*/
#pragma once

// Headers.
#include <vector>
#include <boost/variant.hpp>

// This contains the list of data types.
#define RUN_ON_ROBOT
#include "sample_data/sample_types.h"

// This allows us to use options.
#include "options.h"

namespace gps_control
{

// Types of data supported for internal data storage.
enum sample_data_format
{
    data_format_bool,
    data_format_uint8,
    data_format_int,
    data_format_double
};

class sample
{
private:
    // Length of sample.
    int T_;
    // sensor data for all time steps.
    // IMPORTANT: data management on the internal data is done manually, be sure to allocate and free as necessary.
    std::vector<void*> internal_data_;
    // sensor metadata: size of each field (in number of entries, not bytes).
    std::vector<int> internal_data_size_;
    // sensor metadata: format of each field.
    std::vector<sample_data_format> internal_data_format_;
    // sensor metadata: additional information about each field.
    std::vector<options_map> meta_data_;
    // Note: state and observation definitions are pairs, where the second entry is how far into the past to go.
    // State definition.
    std::vector<std::pair<data_type,int> > state_definition_;
    // Observation definition.
    std::vector<std::pair<data_type,int> > obs_definition_;
public:
    // Constructor.
    sample(int T);
    // Construct state from message.
    sample(gps_control::state_msg::ConstPtr &msg);
    // Destructor.
    virtual ~sample();
    // Get pointer to internal data for given time step.
    virtual void *get_data_pointer(int t, data_type sensor);
    // Add sensor data for given timestep.
    virtual void set_data(int t, data_type sensor, const void *data, int data_size, sample_data_format data_format);
    // Get sensor data for given timestep.
    virtual void get_data(int t, data_type sensor, void *data, int data_size, sample_data_format data_format) const;
    // Set sensor meta-data. Note that this resizes any fields that don't match the current format and deletes their data!
    virtual void set_meta_data(data_type sensor, int data_size, sample_data_format data_format, options_map meta_data_);
    // Get sensor meta-data.
    virtual void get_meta_data(data_type sensor, int &data_size, sample_data_format &data_format, options_map &meta_data_) const;
    // Get the state representation.
    virtual void get_state(int t, Eigen::VectorXd &x) const;
    // Get the observation.
    virtual void get_obs(int t, Eigen::VectorXd &obs) const;
    // Get the action.
    virtual void get_action(int, Eigen::VectorXd &u) const;
};

}


/*
Here is how I think the state assembler should work at a high level:
1. take in a list of things to include in the state, e.g.
"joint angles"
"joint angle velocities"
"previous joint angles"
"previous joint angle velocities"
"FK"
"FK velocities"

2. Given sensors from the robot (see controllers dir), assemble X and phi using this list, but store *all* sensors unless explicitly told not to (e.g. to avoid storing images)

3. Some functionality to be able to take the full sensor state and assemble X and phi on demand, in case we change the state representation later...
*/

// TODO: create an enum here that contains all of the available sensors.
// To create a new sensor:
// 1. add it to the enum
// 2. create corresponding sensor object
// 3. Add a call to the constructor of this object in the robot_plugin
// This should be sufficient, because the number of sensors will be obtained
// automatically from this enum, and the state assembler will automatically
// assemble the state (X and phi) from what is in these enums.
// Each sensor will always have a little bit of meta-data defined, which
// might include things like the dimensions of the image. This meta-data
// needs to be passed around along with the sample... should figure out
// a good way to do this.

