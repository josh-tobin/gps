#include "agent/controller/encodersensor.h"

using namespace gps_control;

/* TODO: need to add Kalman filter, set up Kalman filter parameters, and configure everything correctly with filter */

// Constructor.
EncoderSensor::EncoderSensor(ros::NodeHandle& n, RobotPlugin *plugin)
{
    // Get current joint angles.
    plugin->get_joint_encoder_readings(previous_angles_);
    // Initialize velocities.
    previous_velocities_.resize(previous_angles_.size(),0.0);
    // Initialize temporaries.
    temp_joint_angles_.resize(previous_angles_.size(),0.0);
    // Set time.
    previous_angles_time_ = plugin->get_current_time();
}

// Destructor.
void EncoderSensor::~EncoderSensor()
{
    // Nothing to do here.
}

// Update the sensor (called every tick).
void EncoderSensor::update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step)
{
    if (is_controller_step)
    {
        // Get new vector of joint angles from plugin.
        plugin->get_joint_encoder_readings(temp_joint_angles_,ArmType.ActiveArm);

        // Compute velocities.
        // Note that we can't assume the last angles are actually from one step ago, so we check first.
        // If they are roughly from one step ago, assume the step is correct, otherwise use actual time.
        double update_time = current_time.toSecs() - previous_angles_time_.toSecs();
        if (fabs(update_time - sensor_step_length_)/sensor_step_length_ >= 0.5 &&
            fabs(update_time - sensor_step_length_)/sensor_step_length_ <= 2.0)
        {
            for (unsigned i = 0; i < previous_velocities_.size(); i++)
                previous_velocities_[i] = (temp_joint_angles_[i] - previous_angles_[i])/sensor_step_length_;
        }
        else
        {
            for (unsigned i = 0; i < previous_velocities_.size(); i++)
                previous_velocities_[i] = (temp_joint_angles_[i] - previous_angles_[i])/(update_time - sensor_step_length_);
        }

        // Move temporaries into the previous joint angles.
        for (unsigned i = 0; i < previous_angles_[i].size(); i++)
            previous_angles_[i] = temp_joint_angles_[i];

        // Update stored time.
        previous_angles_time_ = current_time;
    }
}

// The settings include the configuration for the Kalman filter.
void EncoderSensor::configure_sensor(const OptionsMap &options)
{
    // Nothing to do here: encoder sensor has no configuration parameters.
}

// Set data format and meta data on the provided sample.
boost::scoped_ptr<Sample> EncoderSensor::set_sample_data_format(boost::scoped_ptr<Sample> sample) const
{
    // Set joint angles size and format.
    OptionsMap joints_metadata;
    sample->set_meta_data(DataType.JointAngle,previous_angles_.size(),SampleDataFormat.DataFormatDouble,joints_metadata);

    // Set joint velocities size and format.
    OptionsMap velocities_metadata;
    sample->set_meta_data(DataType.JointVelocity,previous_velocities_.size(),SampleDataFormat.DataFormatDouble,joints_metadata);
}

// Set data on the provided sample.
boost::scoped_ptr<Sample> EncoderSensor::set_sample_data(boost::scoped_ptr<Sample> sample) const
{
    // Set joint angles.
    sample->set_data(0,DataType.JointAngle,&previous_angles_[0],previous_angles_.size(),SampleDataFormat.DataFormatDouble);

    // Set joint velocities.
    sample->set_data(0,DataType.JointAngle,&previous_velocities_[0],previous_velocities_.size(),SampleDataFormat.DataFormatDouble);
}
