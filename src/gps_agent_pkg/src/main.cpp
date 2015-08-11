#include <ros/ros.h>
#include <pr2_controllers_msgs/JointTrajectoryAction.h>
#include <actionlib/client/simple_action_client.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Empty.h>
#include <boost/scoped_ptr.hpp>
#include <vector>
#include <string>

#include "gps_agent_pkg/positioncontroller.h"

int main(int argc, char** argv) {
  // Announce this program to the ROS master as a "node" called "hello_world_node"
  ros::init(argc, argv, "hello_world_node");
  // Start the node resource managers (communication, time, etc)
  ros::start();
  // Broadcast a simple log message
  ROS_INFO_STREAM("Hello, world!");
  // Process ROS callbacks until receiving a SIGINT (ctrl-c)
  gps_control::PositionController funsies();
  ros::spin();
  // Stop the node's resources
  ros::shutdown();
  // Exit tranquilly
  return 0;

}
