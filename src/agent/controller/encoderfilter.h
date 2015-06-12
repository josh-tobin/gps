/*
Kalman filter for encoder data. Each filter filters one degree of freedom at
a time.
*/
#pragma once

namespace GPSControl
{

class EncoderFilter
{
private:
    // This should contain Kalman filter settings (precomputed matrices).
    // This should also contain the Kalman filter state information.
public:
    // Constructor.
    EncoderFilter(ros::NodeHandle& n);
    // Destructor.
    virtual ~EncoderFilter();
    // Update the Kalman filter.
    virtual void update(double sec_elapsed, std::vector<double> state);
    // Configure the Kalman filter.
    virtual void configure(/* TODO: decide how to do this part */);
    // Return filtered state and velocity.
    virtual void get_state(double &state, double &velocity);
};

}
