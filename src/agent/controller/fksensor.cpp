#include "agent/controller/fksensor.h"

using namespace gps_control;

// Constructor.
FKSensor::FKSensor(ros::NodeHandle& n, RobotPlugin *plugin)
{
    // Get current joint angles to determine dimensions.
    plugin->get_joint_encoder_readings(temp_joint_angles_,ArmType.ActiveArm);
    unsigned num_joints = temp_joint_angles_.size();

    // Resize KDL joint array.
    temp_joint_array_.resize(num_joints);

    // Resize Jacobian.
    previous_jacobian_.resize(6,num_joints);

    // Allocate space for end effector points (default number is 3).
    previous_end_effector_points_.resize(3,3);
    previous_end_effector_point_velocities_.resize(3,3);
    temp_end_effector_points_.resize(3,3);
    end_effector_points_.resize(3,3);

    // Set time.
    previous_pose_time_ = -1.0; // This ignores the velocities on the first step.
}

// Destructor.
void FKSensor::~FKSensor()
{
    // Nothing to do here.
}

// Update the sensor (called every tick).
void FKSensor::update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step)
{
    if (is_controller_step)
    {
        // Get FK solvers from plugin.
        boost::scoped_ptr<KDL::ChainFkSolverPos> fk_solver;
        boost::scoped_ptr<KDL::ChainJntToJacSolver> jac_solver;
        plugin->get_fk_solver(fk_solver,jac_solver,ArmType.ActiveArm);

        // Get joint angles from encoder sensor (this applies the Kalman filter).
        // TODO: implement

        // Compute end effector position, rotation, and Jacobian.
        // Save angles in KDL joint array.
        for (unsigned i = 0; i < temp_joint_angles_.size(); i++)
            temp_joint_array_(i) = temp_joint_angles_[i];
        // Run the solvers.
        fk_solver_->JntToCart(temp_joint_array_, temp_tip_pose_);
        jac_solver_->JntToJac(temp_joint_array_, temp_jacobian_);
        // Store position, rotation, and Jacobian.
        for (unsigned i = 0; i < 3; i++)
            previous_position_(i) = temp_tip_pose_.p(i);
        for (unsigned j = 0; j < 3; j++)
            for (unsigned i = 0; i < 3; i++)
                previous_rotation(i,j) = temp_tip_pose.M(i,j);
        for (unsigned j = 0; j < temp_jacobian_.columns(); j++)
            for (unsigned i = 0; i < 6; i++)
                previous_jacobian(i,j) = temp_jacobian_(i,j);

        // Compute current end effector points and store in temporary storage.
        temp_end_effector_points_ = previous_rotation_*end_effector_points_;
        temp_end_effector_points_.colwise() += previous_position_;

        /* TODO: very important: remember to adjust for target points! probably best to do this *after* velocity computation in case config changes... */

        // Compute velocities.
        double update_time = current_time.toSecs() - previous_pose_time_.toSecs();
        if (previous_angles_time_ >= 0.0)
        { // Only compute velocities if we have a previous sample.
            if (fabs(update_time)/sensor_step_length_ >= 0.5 &&
                fabs(update_time)/sensor_step_length_ <= 2.0)
            {
                previous_end_effector_velocities_ = (temp_end_effector_points_ - previous_end_effector_points_)/sensor_step_length;
            }
            else
            {
                previous_end_effector_velocities_ = (temp_end_effector_points_ - previous_end_effector_points_)/update_time;
            }
        }

        // Move temporaries into the previous end effector position.
        previous_end_effector_points_ = temp_end_effector_points_;

        // Update stored time.
        previous_pose_time_ = current_time;
    }
}

// Configure the sensor (for sensor-specific trial settings).
// This function is used to pass the end-effector points.
void FKSensor::configure_sensor(const OptionsMap &options)
{
    /* TODO: note that this will get called every time there is a report, so
    we should not throw out the previous transform just because we are trying
    to set end-effector points. Instead, just use the stored transform to
    compute what the points should be! This will allow us to query positions
    and velocities each time. */
}

// Set data format and meta data on the provided sample.
boost::scoped_ptr<Sample> FKSensor::set_sample_data_format(boost::scoped_ptr<Sample> sample) const
{
    // Set end effector point size and format.
    OptionsMap eep_metadata;
    sample->set_meta_data(DataType.EndEffectorPoint,previous_end_effector_points_.cols()*previous_end_effector_points_.rows(),SampleDataFormat.DataFormatDouble,eep_metadata);

    // Set end effector point velocities size and format.
    OptionsMap eepv_metadata;
    sample->set_meta_data(DataType.EndEffectorPointVelocity,previous_end_effector_point_velocities_.cols()*previous_end_effector_point_velocities_.rows(),SampleDataFormat.DataFormatDouble,eepv_metadata);

    // Set end effector position size and format.
    OptionsMap eepos_metadata;
    sample->set_meta_data(DataType.EndEffectorPosition,3,SampleDataFormat.DataFormatDouble,eepos_metadata);

    // Set end effector rotation size and format.
    OptionsMap eerot_metadata;
    sample->set_meta_data(DataType.EndEffectorPosition,9,SampleDataFormat.DataFormatDouble,eerot_metadata);

    // Set jacobian size and format.
    OptionsMap eejac_metadata;
    sample->set_meta_data(DataType.EndEffectorPosition,previous_jacobian_.cols()*previous_jacobian_.rows(),SampleDataFormat.DataFormatDouble,eejac_metadata);
}

// Set data on the provided sample.
boost::scoped_ptr<Sample> FKSensor::set_sample_data(boost::scoped_ptr<Sample> sample) const
{
    // Set end effector point.
    sample->set_data(0,DataType.EndEffectorPoint,previous_end_effector_points_.data(),previous_end_effector_points_.cols()*previous_end_effector_points_.rows(),SampleDataFormat.DataFormatDouble);

    // Set end effector point velocities.
    sample->set_data(0,DataType.EndEffectorPointVelocity,previous_end_effector_point_velocities_.data(),previous_end_effector_point_velocities_.cols()*previous_end_effector_point_velocities_.rows(),SampleDataFormat.DataFormatDouble);

    // Set end effector position.
    sample->set_data(0,DataType.EndEffectorPosition,previous_position_.data(),3,SampleDataFormat.DataFormatDouble);

    // Set end effector rotation.
    sample->set_data(0,DataType.EndEffectorRotation,previous_rotation_.data(),9,SampleDataFormat.DataFormatDouble);

    // Set end effector jacobian.
    sample->set_data(0,DataType.EndEffectorJacobian,previous_jacobian_.data(),previous_jacobian_.cols()*previous_jacobian_.rows(),SampleDataFormat.DataFormatDouble);
}
