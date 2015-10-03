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
#include "gps/proto/gps.pb.h"

// This allows us to use options.
#include "options.h"

namespace gps_control
{

// Types of data supported for internal data storage.
enum SampleDataFormat
{
    SampleDataFormatBool,
    SampleDataFormatUInt8,
    SampleDataFormatInt,
    SampleDataFormatDouble,
    SampleDataFormatEigenMatrix,
    SampleDataFormatEigenVector
};

typedef boost::variant<bool,uint8_t,std::vector<int>,int,double,Eigen::MatrixXd,Eigen::VectorXd> SampleVariant;
typedef std::vector<SampleVariant> SampleList;
typedef std::map<gps::SampleType, SampleList> SampleMap;

class Sample
{
private:
    // Length of sample.
    int T_;
    // sensor data for all time steps.
    // IMPORTANT: data management on the internal data is done manually, be sure to allocate and free as necessary.
    SampleMap internal_data_;
    // sensor metadata: size of each field (in number of entries, not bytes).
    std::vector<int> internal_data_size_;
    // sensor metadata: format of each field.
    std::vector<SampleDataFormat> internal_data_format_;
    // sensor metadata: additional information about each field.
    std::vector<OptionsMap> meta_data_;
    // Note: state and observation definitions are pairs, where the second entry is how far into the past to go.
    // State definition.
    std::vector<std::pair<gps::SampleType,int> > state_definition_;
    // Observation definition.
    std::vector<std::pair<gps::SampleType,int> > obs_definition_;
public:
    // Constructor.
    Sample(int T);
    // Construct state from message.
    //Sample(gps_control::state_msg::ConstPtr& msg); // TODO: Replace with initialization from protobuf
    // Destructor.
    virtual ~Sample();
    // Get pointer to internal data for given time step.
    virtual void *get_data_pointer(int t, gps::SampleType type);
    // Add sensor data for given timestep.
    virtual void set_data(int t, gps::SampleType type, SampleVariant data, int data_size, SampleDataFormat data_format);
    // Get sensor data for given timestep.
    virtual void get_data(int t, gps::SampleType type, void *data, int data_size, SampleDataFormat data_format) const;
    // Set sensor meta-data. Note that this resizes any fields that don't match the current format and deletes their data!
    virtual void set_meta_data(gps::SampleType type, int data_size, SampleDataFormat data_format, OptionsMap meta_data_);
    // Get sensor meta-data.
    virtual void get_meta_data(gps::SampleType type, int &data_size, SampleDataFormat &data_format, OptionsMap &meta_data_) const;
    // Get the state representation.
    virtual void get_state(int t, Eigen::VectorXd &x) const;
    // Get the observation.
    virtual void get_obs(int t, Eigen::VectorXd &obs) const;
    // Fill data arbitrary sensor information
    virtual void get_data(int t, Eigen::VectorXd &data, std::vector<gps::SampleType> datatypes);
    // Get the action.
    virtual void get_action(int, Eigen::VectorXd &u) const;
};

}
