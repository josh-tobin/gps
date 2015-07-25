#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/trialcontroller.h"
#include "gps_agent_pkg/ArmType.h"

using namespace gps_control;

// Constructor.
TrialController::TrialController()
: Controller()
{
    // Set initial time.
    last_update_time_ = ros::Time(0.0);
}

// Destructor.
TrialController::~TrialController()
{
}

// Update the controller (take an action).
void TrialController::update(RobotPlugin *plugin, ros::Time current_time, boost::scoped_ptr<Sample>& sample, Eigen::VectorXd &torques)
{
    Eigen::VectorXd X, obs;
    //TODO: Fill in X and obs from sample

    // Ask subclass to fill in torques
    get_action(step_counter_, X, obs, torques);

    // Update last update time.
    last_update_time_ = current_time;
    step_counter_ ++;
}

void TrialController::configure_controller(OptionsMap &options)
{
    std::vector<int> datatypes;

    datatypes = boost::get<std::vector<int>>(options["state_datatypes"]);
    state_datatypes_.resize(datatypes.size());
    for(int i=0; i<datatypes.size(); i++){
        state_datatypes_[i] = (gps::SampleType) datatypes[i];
    }

    datatypes = boost::get<std::vector<int>>(options["obs_datatypes"]);
    obs_datatypes_.resize(datatypes.size());
    for(int i=0; i<datatypes.size(); i++){
        obs_datatypes_[i] = (gps::SampleType) datatypes[i];
    }

}

// Check if controller is finished with its current task.
bool TrialController::is_finished() const
{
    // Check whether we are close enough to the current target.
    // TODO: implement.
    return true;
}

// Ask the controller to return the sample collected from its latest execution.
boost::scoped_ptr<Sample>* TrialController::get_sample() const
{
    // Return the sample that has been recorded so far.
    // TODO: implement.
    return NULL;
}

// Reset the controller -- this is typically called when the controller is turned on.
void TrialController::reset(ros::Time time)
{
    last_update_time_ = time;
    step_counter_ = 0;
}
