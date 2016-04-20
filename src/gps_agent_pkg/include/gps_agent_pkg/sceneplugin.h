/*
 * This is the class that takes care of interfacing with all of ROS except
 * the robot
 */
#pragma once

// Headers.
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <ros/ros.h>
#include <std_msgs/Empty.h>
#include <std_msgs/Int32.h>
#include <std_srvs/Empty.h>

#include "gps/proto/gps.pb.h"
#include "gazebo_msgs/SpawnModel.h"
#include "gazebo_msgs/DeleteModel.h"
#include "gazebo_msgs/SpawnModel.h"
#include "gazebo_msgs/GetModelState.h"
#include "gps_agent_pkg/SetSceneConfig.h"
#include "gps_agent_pkg/SetModelConfig.h"
#include "pr2_mechanism_msgs/SwitchController.h"

namespace gps_control
{

class ScenePlugin
{
    protected:
        // Name of plugin to stop
        std::string pr2_plugin_;
        // Node handle
        ros::NodeHandle n;
        // Subscribe to topic to know when to modify scene
        ros::Subscriber modify_scene_subscriber_;
        // Service clients
        // request position/orientation of an object in the scene
        ros::ServiceClient model_pos_client_;
        // pause/unpause the physics for constructing / destructing objects
        ros::ServiceClient pause_physics_client_;
        ros::ServiceClient unpause_physics_client_;
        // delete and spawn models
        ros::ServiceClient delete_model_client_;
        ros::ServiceClient spawn_model_client_;
        // stop and start the gps controller
        ros::ServiceClient switch_controller_client_;
        ros::Publisher pub;
    public:
        // Constructor
        ScenePlugin();
        // Destructor
        virtual ~ScenePlugin();
        // Initialize everything
        virtual void initialize();
        // Initialize the ROS subscribers and publishers
        virtual void initialize_ros();
        // Modify scene callback
        virtual void modify_scene_subscriber_callback(const gps_agent_pkg::SetSceneConfig::ConstPtr& msg);
};

}
        
