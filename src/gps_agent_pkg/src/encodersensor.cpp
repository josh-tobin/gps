#include "gps_agent_pkg/encodersensor.h"
#include "gps_agent_pkg/robotplugin.h"

using namespace gps_control;

/* TODO: need to add Kalman filter, set up Kalman filter parameters, and configure everything correctly with filter */

// Constructor.
EncoderSensor::EncoderSensor(ros::NodeHandle& n, RobotPlugin *plugin): Sensor(n, plugin)
{
    // Get current joint angles.
    ROS_INFO_STREAM("beginning constructor");
    plugin->get_joint_encoder_readings(previous_angles_, TrialArm);

    // Initialize velocities.
    previous_velocities_.resize(previous_angles_.size());

    // Initialize temporary angles.
    temp_joint_angles_.resize(previous_angles_.size());

    // Resize KDL joint array.
    temp_joint_array_.resize(previous_angles_.size());

    // Resize Jacobian.
    previous_jacobian_.resize(6,previous_angles_.size());
    temp_jacobian_.resize(previous_angles_.size());

    // Allocate space for end effector points
    previous_end_effector_points_.resize(1,3);
    previous_end_effector_point_velocities_.resize(1,3);
    temp_end_effector_points_.resize(1,3);
    end_effector_points_.resize(1,3);

    // Resize point jacobians
    point_jacobians_.resize(3, previous_angles_.size());
    point_jacobians_rot_.resize(3, previous_angles_.size());


    // Set time.
    previous_angles_time_ = ros::Time(0.0); // This ignores the velocities on the first step.
    ROS_INFO_STREAM("ending constructor");
}

// Destructor.
EncoderSensor::~EncoderSensor()
{
    // Nothing to do here.
}

// Update the sensor (called every tick).
void EncoderSensor::update(RobotPlugin *plugin, ros::Time current_time, bool is_controller_step)
{
    //ROS_INFO_STREAM("EncoderSensor::update");
    if (is_controller_step)
    {
        // Get new vector of joint angles from plugin.
        plugin->get_joint_encoder_readings(temp_joint_angles_, TrialArm);

        // TODO: use Kalman filter...

        // Get FK solvers from plugin.
        plugin->get_fk_solver(fk_solver_,jac_solver_,TrialArm);

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
                previous_rotation_(i,j) = temp_tip_pose_.M(i,j);
        for (unsigned j = 0; j < temp_jacobian_.columns(); j++)
            for (unsigned i = 0; i < 6; i++)
                previous_jacobian_(i,j) = temp_jacobian_(i,j);

        // IMPORTANT: note that the Python code will assume that the Jacobian is the Jacobian of the end effector points, not of the end
        // effector itself. In the old code, this correction was done in Matlab, but since the simulator will produce Jacobians of end
        // effector points directly, it would make sense to also do this transformation on the robot, and send back N Jacobians, one for
        // each feature point.
        // Compute jacobian
        unsigned n_actuator = previous_joint_angles_.size(); //TODO: Assuming we are using all joints
        ROS_INFO("n_actuator: %d", n_actuator);
        for(unsigned i=0; i<n_sites_; i++){
            ROS_INFO("Setting point %d", i);
            unsigned site_start = i*3;
            unsigned site_end = (i+1)*3;
            Eigen::VectorXd ovec = end_effector_points_.row(i);
            ROS_INFO("Site vec= (%d,%d,%d)", site_vec[0], site_vec[1], site_vec[2]);
            
            //point_jacobians_[site_start:site_end, iq] = temp_jacobian_[0:3,:]
            //point_jacobians_rot_[site_start:site_end, iq] = temp_jacobian_[3:6,:]
            for(unsigned j=0; j<3; j++){
                for(unsigned k=0; k<n_actuator; k++){
                    point_jacobians_(site_start+j, k) = temp_jacobian_(j,k)
                    point_jacobians_rot_(site_start+j, k) = temp_jacobian_(j+3,k)
                }
            }
            ROS_INFO("Copied jac");
            
            // Compute site Jacobian.
            ovec = previous_rotation*ovec;

            for(unsigned j=0; j<3; j++){
                point_jacobians_[site_start:site_end, iq]
                MatrixXd Jr;
                Jr = point_jacobians_rot_;
                for(int k=0; k<n_actuator; k++){
                    point_jacobians_(site_start, k) = Jr(site_start+1, k)*ovec[2] - Jr(site_start+2, k)*ovec[1];
                    point_jacobians_(site_start+1, k) = Jr(site_start+2, k)*ovec[0] - Jr(site_start, k)*ovec[2];
                    point_jacobians_(site_start+2, k) = Jr(site_start, k)*ovec[1] - Jr(site_start+1, k)*ovec[0];
                }
                /*
                np.c_[Jr[site_start+1, iq]*ovec[2] - Jr[site_start+2, iq]*ovec[1] , 
                 Jr[site_start+2, iq]*ovec[0] - Jr[site_start, iq]*ovec[2] , 
                 Jr[site_start, iq]*ovec[1] - Jr[site_start+1, iq]*ovec[0]].T
                */
            }
        }

        // Compute current end effector points and store in temporary storage.
        temp_end_effector_points_ = previous_rotation_*end_effector_points_;
        temp_end_effector_points_.colwise() += previous_position_;

        // TODO: very important: remember to adjust for target points! probably best to do this *after* velocity computation in case config changes...

        // Compute velocities.
        // Note that we can't assume the last angles are actually from one step ago, so we check first.
        // If they are roughly from one step ago, assume the step is correct, otherwise use actual time.

        double update_time = current_time.toSec() - previous_angles_time_.toSec();
        if (!previous_angles_time_.isZero())
        { // Only compute velocities if we have a previous sample.
            if (fabs(update_time)/sensor_step_length_ >= 0.5 &&
                fabs(update_time)/sensor_step_length_ <= 2.0)
            {
                previous_end_effector_point_velocities_ = (temp_end_effector_points_ - previous_end_effector_points_)/sensor_step_length_;
                for (unsigned i = 0; i < previous_velocities_.size(); i++){
                    previous_velocities_[i] = (temp_joint_angles_[i] - previous_angles_[i])/sensor_step_length_;
                }
            }
            else
            {
                previous_end_effector_point_velocities_ = (temp_end_effector_points_ - previous_end_effector_points_)/update_time;
                for (unsigned i = 0; i < previous_velocities_.size(); i++){
                    previous_velocities_[i] = (temp_joint_angles_[i] - previous_angles_[i])/update_time;
                }
            }
        }

        // Move temporaries into the previous joint angles.
        previous_end_effector_points_ = temp_end_effector_points_;
        for (unsigned i = 0; i < previous_angles_.size(); i++){
            previous_angles_[i] = temp_joint_angles_[i];
        }

        // Update stored time.
        previous_angles_time_ = current_time;
    }
}

// The settings include the configuration for the Kalman filter.
void EncoderSensor::configure_sensor(OptionsMap &options)
{
    // TODO: should set up Kalman filter here.
    /* TODO: note that this will get called every time there is a report, so
    we should not throw out the previous transform just because we are trying
    to set end-effector points. Instead, just use the stored transform to
    compute what the points should be! This will allow us to query positions
    and velocities each time. */
    ROS_WARNING("Kalman filter configuration not implemented!");

    end_effector_points_ = boost::get<Eigen::MatrixXd>(options["ee_sites"]);
    n_points_ = end_effector_points_.rows()
    if( end_effector_points_.cols() != 3){
        ROS_ERROR("EE Sites have more than 3 coordinates: Shape=(%d,%d)", n_points_,
                end_effector_points_.cols());
    }
    previous_end_effector_points_.resize(n_points_,3);
    previous_end_effector_point_velocities_.resize(n_points_,3);
    temp_end_effector_points_.resize(n_points_,3);
    end_effector_points_.resize(n_points_,3);
    point_jacobians_.resize(3*n_points_, previous_angles_.size());
    point_jacobians_rot_.resize(3*n_points_, previous_angles_.size());

}

// Set data format and meta data on the provided sample.
void EncoderSensor::set_sample_data_format(boost::scoped_ptr<Sample>& sample)
{
    // Set joint angles size and format.
    OptionsMap joints_metadata;
    sample->set_meta_data(gps::JOINT_ANGLES,previous_angles_.size(),SampleDataFormatEigenVector,joints_metadata);

    // Set joint velocities size and format.
    OptionsMap velocities_metadata;
    sample->set_meta_data(gps::JOINT_VELOCITIES,previous_velocities_.size(),SampleDataFormatEigenVector,joints_metadata);

    // Set end effector point size and format.
    OptionsMap eep_metadata;
    sample->set_meta_data(gps::END_EFFECTOR_POINTS,previous_end_effector_points_.cols()*previous_end_effector_points_.rows(),SampleDataFormatEigenVector,eep_metadata);

    // Set end effector point velocities size and format.
    OptionsMap eepv_metadata;
    sample->set_meta_data(gps::END_EFFECTOR_POINT_VELOCITIES,previous_end_effector_point_velocities_.cols()*previous_end_effector_point_velocities_.rows(),SampleDataFormatEigenVector,eepv_metadata);

    // Set end effector position size and format.
    OptionsMap eepos_metadata;
    sample->set_meta_data(gps::END_EFFECTOR_POSITIONS,3,SampleDataFormatEigenVector,eepos_metadata);

    // Set end effector rotation size and format.
    OptionsMap eerot_metadata;
    sample->set_meta_data(gps::END_EFFECTOR_ROTATIONS,9,SampleDataFormatEigenMatrix,eerot_metadata);

    // Set jacobian size and format.
    OptionsMap eejac_metadata;
    sample->set_meta_data(gps::END_EFFECTOR_JACOBIANS,previous_jacobian_.cols()*previous_jacobian_.rows(),SampleDataFormatEigenMatrix,eejac_metadata);
}

// Set data on the provided sample.
void EncoderSensor::set_sample_data(boost::scoped_ptr<Sample>& sample, int t)
{
    // Set joint angles.
    sample->set_data(t,gps::JOINT_ANGLES,previous_angles_,previous_angles_.size(),SampleDataFormatEigenVector);

    // Set joint velocities.
    sample->set_data(t,gps::JOINT_VELOCITIES,previous_velocities_,previous_velocities_.size(),SampleDataFormatEigenVector);


    // Set end effector point.
    // Flatten points - maybe this should be kept as a matrix?
    Eigen::VectorXd flattened_ee_pts = previous_end_effector_points_;
    flattened_ee_pts.resize(previous_end_effector_points_.cols()*previous_end_effector_points_.rows(), 1);
    sample->set_data(t,gps::END_EFFECTOR_POINTS,flattened_ee_pts,previous_end_effector_points_.cols()*previous_end_effector_points_.rows(),SampleDataFormatEigenVector);

    // Set end effector point velocities.
    //Flatten velocities - maybe this should be kept as a matrix?
    Eigen::VectorXd flattened_ee_vel = previous_end_effector_point_velocities_;
    flattened_ee_vel.resize(previous_end_effector_point_velocities_.cols()*previous_end_effector_point_velocities_.rows(), 1);
    sample->set_data(t,gps::END_EFFECTOR_POINT_VELOCITIES,flattened_ee_vel,previous_end_effector_point_velocities_.cols()*previous_end_effector_point_velocities_.rows(),SampleDataFormatEigenVector);

    // Set end effector position.
    Eigen::VectorXd flattened_position; //Need to convert Vector3d to VectorXd. Eigen seems finicky about this.
    flattened_position.resize(3, 1);
    for (unsigned i = 0; i < 3; i++)
        flattened_position[i] = previous_position_[i];
    sample->set_data(t,gps::END_EFFECTOR_POSITIONS,flattened_position,3,SampleDataFormatEigenVector);

    // Set end effector rotation.
    Eigen::MatrixXd new_rot; //Need to convert Matrix3d to MatrixXd. Eigen seems finicky about this.
    new_rot.resize(3, 3);
    for (unsigned i = 0; i < 3; i++)
        for (unsigned j = 0; j < 3; j++)
            new_rot(i,j) = previous_rotation_(i,j);
    sample->set_data(t,gps::END_EFFECTOR_ROTATIONS,new_rot,9,SampleDataFormatEigenMatrix);

    // Set end effector jacobian.
    sample->set_data(t,gps::END_EFFECTOR_JACOBIANS,previous_jacobian_,previous_jacobian_.cols()*previous_jacobian_.rows(),SampleDataFormatEigenMatrix);
}
