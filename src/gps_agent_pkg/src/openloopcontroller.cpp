#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/openloopcontroller.h"
#include "gps_agent_pkg/util.h"

using namespace gps_control;

// Constructor
OpenLoopController::OpenLoopController()
: TrialController()
{
    is_configured_ = false;
}

// Destructor
OpenLoopController::~OpenLoopController()
{
}

void OpenLoopController::get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U)
{
    U = U_[t];
}

//Configure the controller
void OpenLoopController::configure_controller(OptionsMap &options)
{
    TrialController::configure_controller(options);

    int T = boost::get<int>(options["T"]);

    U_.resize(T);
    for(int i=0; i<T; i++){
        U_[i] = boost::get<Eigen::VectorXd>(options["U_"+to_string(i)]);
    }
    ROS_INFO_STREAM("Configured open loop controller");
    is_configured_ = true;
}
