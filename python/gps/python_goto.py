import imp
import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
from gps import __file__ as gps_filepath
import numpy as np
import rospy
from gps.agent.agent import Agent
from gps.agent.ros.agent_ros import AgentROS
from gps.agent.config import AGENT_ROS
from gps.agent.ros.ros_utils import ServiceEmulator, msg_to_sample, \
				    policy_to_msg
from gps.proto.gps_pb2 import TRIAL_ARM, AUXILIARY_ARM
import argparse
from gps_agent_pkg.msg import DataRequest

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('position', type=float, nargs='+')
    args = parser.parse_args()
    

    hyperparams_file = 'experiments/pr2_example/hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    agent = AgentROS(hyperparams.config['agent'])
    condition_data = agent._hyperparams['reset_conditions'][0]
    target_position = np.array(args.position)    
    #print 'Target position is: ' + str(target_position)
    agent.reset_arm(TRIAL_ARM, condition_data[TRIAL_ARM]['mode'], 
                    target_position)
    request = DataRequest()
    request.id = agent._get_next_seq_id()
    request.arm = TRIAL_ARM
    request.stamp = rospy.get_rostime()
    result_msg = agent._data_service.publish_and_wait(request)
    #print type(result_msg)
    print result_msg.sensor_data[7].data
if __name__ == '__main__':
    main()
