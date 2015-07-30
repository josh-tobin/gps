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
    U = K_[t]*X+k_[t];
}

// Configure the controller.
void LinearGaussianController::configure_controller(OptionsMap &options)
{
    // TODO: Update K_
    int T = boost::get<int>(options["T"]);

    K_.resize(T);
    for(int i=0; i<T; i++){
        K_[i] = boost::get<Eigen::MatrixXd>(options["K_"+std::to_string(i)]);
    }

    k_.resize(T);
    for(int i=0; i<T; i++){
        k_[i] = boost::get<Eigen::MatrixXd>(options["k_"+std::to_string(i)]);
    }
}
