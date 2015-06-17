#include "agent/controller/robotplugin.h"
#include "agent/controller/positioncontroller.h"

using namespace gps_control;

// Constructor.
PositionController::PositionController(ros::NodeHandle& n, ArmType arm)
{
    // Initialize PD gains.
    

    // Initialize velocity bounds.


    // Initialize integral terms to zero.
    

    // Initialize current angle and position.
    

    // Initialize target angle and position.
    

    // Initialize joints temporary storage.
    

    // Initialize Jacobian temporary storage.
    temp_jacobian_.resize(6,current_angles_.rows());

    // Set initial mode.
    mode_ = PositionControlMode.NoControl;

    // Set initial time.
    last_update_time = ros::Time(0.0);

    // Set arm.
    arm_ = arm;
}

// Destructor.
void PositionController::~PositionController()
{
}

// Update the controller (take an action).
void PositionController::update(RobotPlugin *plugin, ros::Time current_time, std::scopted_ptr<Sample> sample, VectorXd &torques)
{
    // Get current joint angles.
    plugin->get_joint_encoder_readings(temp_angles_,arm_);

    // Check dimensionality.
    assert(temp_angles_.rows() == torques_.rows());
    assert(temp_angles_.rows() == current_angles_.rows());

    // Estimate joint angle velocities.
    double update_time = current_time.toSecs() - last_update_time_.toSecs();
    if (!last_update_time_.isZero())
    { // Only compute velocities if we have a previous sample.
        current_angle_velocities_ = (temp_angles_ - current_angles_)/update_time;
    }

    // Store new angles.
    current_angles_ = temp_angles_;

    // Update last update time.
    last_update_time_ = current_time;

    // If doing task space control, compute joint positions target.
    if (mode_ == PositionControlMode.TaskSpaceControl)
    {
        ROS_ERROR("Not implemented!");

        // Get current end effector position.
        // TODO: implement.

        // Get current Jacobian.
        // TODO: implement.

        // TODO: should also try Jacobian pseudoinverse, it may work a little better.
        // Compute desired joint angle offset using Jacobian transpose method.
        target_angles_ = current_angles_ + temp_jacobian_.transpose()*(target_pose_ - current_pose_);
    }

    // If we're doing any kind of control at all, compute torques now.
    if (mode_ != PositionControlMode.NoControl)
    {
        // Compute error.
        temp_angles_ = current_angles_ - target_angles_;

        // Add to integral term.
        pd_integral_ += temp_angles;

        // Compute torques.
        // TODO: look at PR2 PD controller implementation and make sure our version matches!
        torques = -((pd_gains_p_.array() * temp_angles_.array()) +
                    (pd_gains_d_.array() * current_angle_velocities_.array()) +
                    (pd_gains_i_.array() * pd_integral_.array())).matrix();
    }

    // TODO: shall we update the stored sample somewhere?
    // TODO: need to decide how we'll deal with samples
    // TODO: shall we just always return the latest sample, or actually accumulate?
    // TODO: might be better to just return the latest one...
}

// Configure the controller.
void PositionController::configure_controller(const OptionsMap &options)
{
    // TODO: implement!
    // This sets the target position.
    // This sets the mode
}

// Check if controller is finished with its current task.
bool PositionController::is_finished() const
{
    // Check whether we are close enough to the current target.
    // TODO: implement.
}

// Ask the controller to return the sample collected from its latest execution.
boost::scoped_ptr<Sample> PositionController::get_sample() const
{
    // Return the sample that has been recorded so far.
    // TODO: implement.
}

// Reset the controller -- this is typically called when the controller is turned on.
void PositionController::reset()
{
    // Clear the integral term.
    pd_integral_.fill(0.0);

    // Clear update time.
    last_update_time = ros::Time(0.0);
}
