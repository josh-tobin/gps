#include "gps_agent_pkg/sceneplugin.h"
#include <urdf/model.h>
#include <fstream>

using namespace gps_control;

// Plugin constructor
ScenePlugin::ScenePlugin()
{
    // Nothing to do here
}

// Destructor
ScenePlugin::~ScenePlugin()
{
    // Nothing to do here
}

// Initialize everything
void ScenePlugin::initialize()
{
    ROS_INFO_STREAM("Initializing ScenePlugin");
    // Keep track of the name of the pr2 plugin. Maybe we should link this
    // to somewhere in the PR2 code.
    pr2_plugin_ = (std::string) "GPSPR2Plugin";
    initialize_ros();

    pub = n.advertise<std_msgs::Int32>("/test_topic", 1);

    std_msgs::Int32 one;
    one.data = 1;
    pub.publish(one);
        
}

// Initialize ROS communication infrastructure
void ScenePlugin::initialize_ros()
{
    ROS_INFO_STREAM("Initializing ROS scene pubs, subs, and services");
    
    // Subscriber that will tell us when to reset objects
    modify_scene_subscriber_ = n.subscribe("/gps_reset_scene", 1, &ScenePlugin::modify_scene_subscriber_callback, this);

    // Create clients that will be used to interact with the ROS scene
    model_pos_client_ = n.serviceClient<gazebo_msgs::GetModelState>("/gazebo/get_model_state");
    pause_physics_client_ = n.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
    unpause_physics_client_ = n.serviceClient<std_srvs::Empty>("gazebo/unpause_physics");
    delete_model_client_ = n.serviceClient<gazebo_msgs::DeleteModel>("/gazebo/delete_model");
    spawn_model_client_ = n.serviceClient<gazebo_msgs::SpawnModel>("/gazebo/spawn_urdf_model");
    switch_controller_client_ = n.serviceClient<pr2_mechanism_msgs::SwitchController>("/pr2_controller_manager/switch_controller");
}

void ScenePlugin::modify_scene_subscriber_callback(const gps_agent_pkg::SetSceneConfig::ConstPtr& msg) {
    
    ROS_INFO_STREAM("received modify scene command");
    std::vector<gps_agent_pkg::SetModelConfig> model_config_commands = msg->model_configs;
    for (int model = 0; model < model_config_commands.size(); model++) {
        std::string model_name = model_config_commands[model].model_name;

        std::string new_model_urdf = (std::string) (model_config_commands[model].new_model_urdf);
        ROS_INFO_STREAM(">>> About to reset state of model " + model_name);
        
        // First, switch off the controller
        pr2_mechanism_msgs::SwitchController switch_cont;
        std::vector<std::string> to_start;
        std::vector<std::string> to_stop;
        to_stop.push_back(pr2_plugin_);
        switch_cont.request.start_controllers = to_start;
        switch_cont.request.stop_controllers = to_stop;
        switch_cont.request.strictness = 1;
        ROS_INFO_STREAM(">>> Pausing controller");
        switch_controller_client_.call(switch_cont);
        // Pause the physics
        std_srvs::Empty empty_msg;
        ROS_INFO_STREAM(">>> Pausing physics");
        pause_physics_client_.call(empty_msg);
        // Get the position of the object
        gazebo_msgs::GetModelState model_state;
        model_state.request.model_name = model_name;
        ROS_INFO_STREAM(">>> Finding model position");
        model_pos_client_.call(model_state);
        // Delete the object
        gazebo_msgs::DeleteModel model_to_delete;
        model_to_delete.request.model_name = model_name;
        ROS_INFO_STREAM(">>> Deleting model");
        delete_model_client_.call(model_to_delete);
       
        // Build the new object
        // to_do: better way to contain urdf package path?
        //std::ifstream ifs((std::string)"/home/jt/gps/src/gps_agent_pkg/urdf/" + new_model_urdf);
        //std::string urdf_content( (std::istreambuf_iterator<char>(ifs)), (std::istreambuf_iterator<char>()   )); 
        
        gazebo_msgs::SpawnModel model_to_spawn;
        
        const char *c = new_model_urdf.c_str(); 
        std::ifstream ifs;
        ifs.open(c);
        std::string line;
        while (!ifs.eof()) // parse the contents of the given urdf in a string
        {
            std::getline(ifs, line);
            model_to_spawn.request.model_xml += line;
        }
        ifs.close();

        model_to_spawn.request.model_name = model_name;
        //model_to_spawn.request.model_xml = urdf_parsed;
        //model_to_spawn.request.model_xml = new_model_urdf;
        //ifs >> model_to_spawn.request.model_xml;
        model_to_spawn.request.initial_pose = model_state.response.pose;
        ROS_INFO_STREAM(">>> Spawning model");
        spawn_model_client_.call(model_to_spawn);
        // Restart the physics
        ROS_INFO_STREAM(">>> Unpausing physics");
        unpause_physics_client_.call(empty_msg);
        // Finally, switch the controller back on
        std::vector<std::string> start_controllers;
        start_controllers.push_back(pr2_plugin_);
        std::vector<std::string> stop_controllers;
        pr2_mechanism_msgs::SwitchController cont;
        cont.request.start_controllers = start_controllers;
        cont.request.stop_controllers = stop_controllers;
        cont.request.strictness = 1;
        ROS_INFO_STREAM(">>> Resetting controller");
        switch_controller_client_.call(cont);
    }
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "scene_plugin");
    ScenePlugin scene_plugin;
    scene_plugin.initialize();
    ros::spin();

    return 0;
}
