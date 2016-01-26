
from framework import *
from math import sqrt
from gps.proto.gps_pb2 import *
import numpy as np

# Original inspired by a contribution by roman_m
# Dimensions scooped from APE (http://www.cove.org/ape/index.htm)
class Crawler (Framework):
    def __init__(self, x0=[0], target=[10]):
        super(Crawler, self).__init__()
       
        self.x0 = x0
        ground = self.world.CreateStaticBody(
                shapes=[ 
                        b2EdgeShape(vertices=[(-50,0),(50,0)]),
                        b2EdgeShape(vertices=[(-50,0),(-50,10)]),
                        b2EdgeShape(vertices=[(50,0),(50,10)]),
                    ]
                ) 
        self.ground_level = 1.01499        
        rectangle_fixture=b2FixtureDef(
                shape=b2PolygonShape(box=(3.0,0.5)),
                density=1.5,
                friction=0.3)

        self.body1 = self.world.CreateDynamicBody(
                fixtures=rectangle_fixture,
                position=((x0[0], self.ground_level)),
                angle=.25*b2_pi
                )
        self.body2=self.world.CreateDynamicBody(
                fixtures=rectangle_fixture,
                position=(x0[0] + 3, self.ground_level),
                angle=-.25*b2_pi
                )
        self.joint=self.world.CreateRevoluteJoint(
                bodyA=self.body1,
                bodyB=self.square,
                localAnchorA=(2.5,0),
                localAnchorB=(-2.5,0)
                enableMotor = True,
                maxMotorTorque=400,
                )
    def Step(self, settings, action):
        self.body.motorSpeed = action[0]

        super(PointMassWorld, self).Step(settings)

    def reset_world(self):
        self.world.ClearForces()
        self.joint.motorSpeed = 0
        self.body1.linearVelocity= (0, 0)
        self.body1.angularVelocity= 0
        self.body2.linearVelocity = (0, 0)
        self.body2.angularVelocity = 0
        self.body1.position = (self.x0[0], self.ground_level),
        self.body1.angle = .25*b2_pi
        self.body2.position = (self.x0[0] + 3, self.ground_level)
        self.body2.angle = -.25*b2_pi


    def get_state(self):
        state = {POSITION : np.array(self.body1.position), JOINT_ANGLES : np.array(self.joint.angle)}

        return state
    
    # def Keyboard(self, key):
    #     if key==Keys.K_a:
    #         self.joint.motorSpeed += 2
    #         print(self.joint.motorSpeed)
    #     elif key==Keys.K_d:
    #         self.joint.motorSpeed-=2
    #     elif key==Keys.K_s:
    #         self.reset_world()

if __name__=="__main__":
     main(Crawler)
