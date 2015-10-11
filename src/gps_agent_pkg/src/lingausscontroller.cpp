#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/lingausscontroller.h"
#include "gps_agent_pkg/utils.h"

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
    U = K_[t]*X+k_[t];
}

// Configure the controller.
void LinearGaussianController::configure_controller(OptionsMap &options)
{
    //Call superclass
    TrialController::configure_controller(options);

    ROS_INFO_STREAM("Received LG parameters");

    // TODO: Update K_
    int T = boost::get<int>(options["T"]);

    //TODO Don't do this hacky string indexing
    K_.resize(T);
    for(int i=0; i<T; i++){
        K_[i] = boost::get<Eigen::MatrixXd>(options["K_"+to_string(i)]);
    }

    k_.resize(T);
    for(int i=0; i<T; i++){
<<<<<<< HEAD
        k_[i] = boost::get<Eigen::VectorXd>(options["k_"+std::to_string(i)]);
=======
        k_[i] = boost::get<Eigen::MatrixXd>(options["k_"+to_string(i)]);
>>>>>>> d4cd787ba9d38ed414893fcedccaf21a76fd7531
    }
}
