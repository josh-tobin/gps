#include "gps_agent_pkg/sample.h"
#include "gps/proto/gps.pb.h"

using namespace gps_control;

Sample::Sample(int T)
{
}

Sample::~Sample()
{
}

void* Sample::get_data_pointer(int t, gps::SampleType type)
{
    return NULL;
}

void Sample::set_data(int t, gps::SampleType type, const void *data, int data_size, SampleDataFormat data_format)
{
    return;
}

void Sample::get_data(int t, gps::SampleType type, void *data, int data_size, SampleDataFormat data_format) const
{
    return;
}

    
void Sample::set_meta_data(gps::SampleType type, int data_size, SampleDataFormat data_format, OptionsMap meta_data_)
{
    return;
}

void Sample::get_meta_data(gps::SampleType type, int &data_size, SampleDataFormat &data_format, OptionsMap &meta_data_) const
{
    return;
}

void Sample::get_state(int t, Eigen::VectorXd &x) const
{
    return;
}

void Sample::get_obs(int t, Eigen::VectorXd &obs) const
{
    return;
}

void Sample::get_action(int, Eigen::VectorXd &u) const
{
    return;
}

