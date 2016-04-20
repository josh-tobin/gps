#!/usr/bin/env python

import sys
import rospy
from pr2_mechanism_msgs.srv import SwitchController
def client():
    print "Waiting for service"
    rospy.wait_for_service('/pr2_controller_manager/switch_controller')
    try:
        print "creating service proxy"
        switch_controller = rospy.ServiceProxy('/pr2_controller_manager/switch_controller', SwitchController)
        #msg = SwitchController()
        #msg.request.start_controllers = []
        #msg.request.stop_controllers = ['GPSPR2Plugin']
        #msg.strictness = 1
        print "calling service"
        switch_controller(start_controllers=[], stop_controllers = ['GPSPR2Plugin'], strictness = 1)
        print "Done"
    except rospy.ServiceException, e:
        print "Service call failed: %s"%e
if __name__ == "__main__":
    client()
