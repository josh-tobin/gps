#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/lingausscontroller.h"

using namespace gps_control;

// Constructor.
LinearGaussianController::LinearGaussianController(ros::NodeHandle& n)
: TrialController(n)
{
}

// Destructor.
LinearGaussianController::~LinearGaussianController()
{
}


void LinearGaussianController::get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U){
    U = K_[t]*X+k_[t];
}

// Configure the controller.
void LinearGaussianController::configure_controller(const OptionsMap &options)
{
    // TODO: Update K_
    //K_;

    // TODO: Update k_
    //k_;
}
