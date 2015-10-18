#include "gps_agent_pkg/positioncontroller.h"
#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/utils.h"

using namespace gps_control;

// Constructor.

// Constructor.
PositionController::PositionController(ros::NodeHandle& n, ArmType arm, int size)
    : Controller(n, arm, size)
{
    // Initialize PD gains.
    pd_gains_p_.resize(size);
    pd_gains_d_.resize(size);
    pd_gains_i_.resize(size);

    // Initialize velocity bounds.
    max_velocities_.resize(size);

    // Initialize integral terms to zero.
    pd_integral_.resize(size);

    // Initialize current angle and position.
    current_angles_.resize(size);
    current_angle_velocities_.resize(size);
    current_pose_.resize(size);

    // Initialize target angle and position.
    target_angles_.resize(size);
    target_pose_.resize(size);

    // Initialize joints temporary storage.
    temp_angles_.resize(size);

    // Initialize torques. TODO: DO WE NEED THIS?
    // torques_.resize(size);

    // Initialize Jacobian temporary storage.
    temp_jacobian_.resize(6,size);
    ROS_INFO_STREAM("jacobian size: " + to_string(temp_jacobian_.size()));

    for (int i = 0; i < size; i++)
    {
        pd_gains_p_[i] = 1.0;
        pd_gains_d_[i] = 0.3;
        pd_gains_i_[i] = 0.02;
        max_velocities_[i] = 3.0;
        //target_angles_[i] = 0.5;
    }
    // Set initial mode.
    mode_ = NoControl;

    // Set initial time.
    last_update_time_ = ros::Time(0.0);

    // Set arm.
    arm_ = arm;
}

// Destructor.
PositionController::~PositionController()
{
}

// Update the controller (take an action).
void PositionController::update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques)
{
    //ROS_INFO_STREAM(">beginning position update");
    // Get current joint angles.
    plugin->get_joint_encoder_readings(temp_angles_,arm_);

    // Check dimensionality.
    assert(temp_angles_.rows() == torques.rows());
    assert(temp_angles_.rows() == current_angles_.rows());

    // Estimate joint angle velocities.
    double update_time = current_time.toSec() - last_update_time_.toSec();
    if (!last_update_time_.isZero())
    { // Only compute velocities if we have a previous sample.
        current_angle_velocities_ = (temp_angles_ - current_angles_)/update_time;
    }

    // Store new angles.
    current_angles_ = temp_angles_;

    // Update last update time.
    last_update_time_ = current_time;

    // If doing task space control, compute joint positions target.
    if (mode_ == TaskSpaceControl)
    {
        ROS_ERROR("Not implemented!");

        // Get current end effector position.
        // TODO: implement.

        // Get current Jacobian.
        // TODO: implement.

        // TODO: should also try Jacobian pseudoinverse, it may work a little better.
        // Compute desired joint angle offset using Jacobian transpose method.
        target_angles_ = current_angles_ + temp_jacobian_.transpose() * (target_pose_ - current_pose_);
    }

    // If we're doing any kind of control at all, compute torques now.
    if (mode_ != NoControl)
    {
        // Compute error.
        temp_angles_ = current_angles_ - target_angles_;

        // Add to integral term.
        pd_integral_ += temp_angles_;

        // Compute torques.
        // TODO: look at PR2 PD controller implementation and make sure our version matches!
        torques = -((pd_gains_p_.array() * temp_angles_.array()) +
                    (pd_gains_d_.array() * current_angle_velocities_.array()) +
                    (pd_gains_i_.array() * pd_integral_.array())).matrix();
        //ROS_INFO_STREAM("joint outputs mode:");
        //ROS_INFO_STREAM(mode_);
        //ROS_INFO_STREAM(torques);
    }
    else
    {
        torques = Eigen::VectorXd::Zero(torques.rows());
        //ROS_INFO_STREAM("joint outputs mode:");
        //ROS_INFO_STREAM(mode_);
        //ROS_INFO_STREAM(torques);
    }

    // TODO: shall we update the stored sample somewhere?
    // TODO: need to decide how we'll deal with samples
    // TODO: shall we just always return the latest sample, or actually accumulate?
    // TODO: might be better to just return the latest one...
}

// Configure the controller.
void PositionController::configure_controller(OptionsMap &options)
{
    // TODO: implement!
    // This sets the target position.
    // This sets the mode
    ROS_INFO_STREAM("Received controller configuration");
    mode_ = (PositionControlMode) boost::get<int>(options["mode"]);
    if (mode_ != NoControl){
        Eigen::VectorXd data = boost::get<Eigen::VectorXd>(options["data"]);
        Eigen::MatrixXd pd_gains = boost::get<Eigen::MatrixXd>(options["pd_gains"]);
        if(mode_ == JointSpaceControl){
            target_angles_ = data;
        }else{
            ROS_ERROR("Unimplemented position control mode!");
        }
    }
}

// Check if controller is finished with its current task.
bool PositionController::is_finished() const
{
    // Check whether we are close enough to the current target.
    // TODO: implement.
    return true;
}
/*

// Ask the controller to return the sample collected from its latest execution.
*/
boost::scoped_ptr<Sample>* PositionController::get_sample() const
{
    // Return the sample that has been recorded so far.
    // TODO: implement.
    return NULL;
}

// Reset the controller -- this is typically called when the controller is turned on.
void PositionController::reset(ros::Time time)
{
    // Clear the integral term.
    pd_integral_.fill(0.0);

    // Clear update time.
    last_update_time_ = ros::Time(0.0);
}

