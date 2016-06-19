
#include "gps_agent_pkg/robotplugin.h"
#include "gps_agent_pkg/util.h"
#include "gps_agent_pkg/lowdofcontroller.h"

using namespace gps_control;

LowDofController::LowDofController(ros::NodeHandle& n, gps::ActuatorType arm,
                                   int size) 
{
    pd_controller_ = new PositionController(n, arm, size);
} 

void LowDofController::get_action(int t, const Eigen::VectorXd &X, const Eigen::VectorXd &obs, Eigen::VectorXd &U){
    if (is_configured_) {
        if(last_command_id_acted_upon < last_command_id_received){
            last_command_id_acted_upon = last_command_id_received;
            failed_attempts = 0;
            pd_controller_->get_action(t, X, obs, U_pd_); 
            // NEED TO ADD: RobotPlugin *plugin, ros::Time current_time,
            // boost::scoped_ptr<Sample>& sample
            //pd_controller_.update(plugin, current_time, sample, U_pd_);
            U_dofs_ = last_action_command_received;
            U = U_pd_;
            for (int i = 0; i < dofs_.size(); i++) {
                U(dofs_[i]) = U_dofs_(i);
            }
        }
        else if(failed_attempts < 2){ //this would allow acting on stale actions...maybe a bad idea?
            U = last_action_command_received;
            failed_attempts++;
        }
        else{
            ROS_FATAL("no new action command received. Can not act on stale actions.");
        }
    }
}

void LowDofController::configure_controller(OptionsMap &options)
{
    ROS_INFO_STREAM("Setting up LowDofController");
    TfController::configure_controller(options);
    pd_controller_->configure_controller(options);
    dofs_ = (std::vector<int>) boost::get< std::vector<int> >(options["dofs"]);
}


