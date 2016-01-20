#include "caffe/caffe.hpp"

#include "gps_agent_pkg/caffenncontroller.h"
#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/util.h"

using namespace gps_control;

// Constructor.
CaffeNNController::CaffeNNController()
: TrialController()
{
    is_configured_ = false;
}

// Destructor.
CaffeNNController::~CaffeNNController()
{
}

void CaffeNNController::get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U){
    if (is_configured_) {
        net_->forward(obs, U);
    }
}

// Configure the controller.
void CaffeNNController::configure_controller(OptionsMap &options)
{
    //Call superclass
    TrialController::configure_controller(options);

    std::string net_param_string = boost::get<string>(options["net_param"]);

    NetParameter net_param;
    net_param.ParseFromString(net_param_string);
    net_.reset(new NeuralNetworkCaffe(net_param));
    net_->set_weights(net_param);

    ROS_INFO_STREAM("Set Caffe network parameters");
    is_configured_ = true;
}
