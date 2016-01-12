#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/lingausscontroller.h"
#include "gps_agent_pkg/util.h"

using namespace gps_control;

// Constructor.
NeuralNetworkController::NeuralNetworkController()
: TrialController()
{
    is_configured_ = false;
}

// Destructor.
NeuralNetworkController::~NeuralNetworkController()
{
}


void NeuralNetworkController::get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U){
}

// Configure the controller.
void NeuralNetworkController::configure_controller(OptionsMap &options)
{
    //Call superclass
    TrialController::configure_controller(options);

    weights_string_ = boost::get<string>(options["weights_string"]);
    model_prototxt_ = boost::get<string>(options["model_prototxt"]);

    // TODO - Construct network here

    ROS_INFO_STREAM("Received Caffe network parameters");
    is_configured_ = true;
}
