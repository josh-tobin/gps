#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/lingausscontroller.h"

using namespace gps_control;

// Constructor.
LinearGaussianController::LinearGaussianController()
: TrialController()
{
}

// Destructor.
LinearGaussianController::~LinearGaussianController()
{
}


void LinearGaussianController::get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U){
    ROS_INFO_STREAM(">>beginning LG update");
    U = k_[t];
    ROS_INFO_STREAM(">>added bias");
    //U += K_[t]*X; //TODO: This crashes
    ROS_INFO_STREAM(">>end LG update");
}

// Configure the controller.
void LinearGaussianController::configure_controller(OptionsMap &options)
{
    //Call superclass
    TrialController::configure_controller(options);

    ROS_INFO_STREAM("Received LG parameters");

    // TODO: Update K_
    int T = boost::get<int>(options["T"]);

    K_.resize(T);
    for(int i=0; i<T; i++){
        K_[i] = boost::get<Eigen::MatrixXd>(options["K_"+std::to_string(i)]);
    }

    k_.resize(T);
    for(int i=0; i<T; i++){
        k_[i] = boost::get<Eigen::VectorXd>(options["k_"+std::to_string(i)]);
    }

    ROS_INFO_STREAM("Finished setting LG parameters");
}
