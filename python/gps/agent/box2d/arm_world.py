
from framework import *
from math import sqrt
from gps.proto.gps_pb2 import *
import numpy as np

# Original inspired by a contribution by roman_m
# Dimensions scooped from APE (http://www.cove.org/ape/index.htm)
class ArmWorld(Framework):
    def __init__(self, x0=[0, 1, 0], target=[10]):
        super(ArmWorld, self).__init__()

        fixture_length = 3
        self.x0 = x0



        rectangle1_fixture=b2FixtureDef(
                shape=b2PolygonShape(box=(.5, fixture_length)),
                density=1.5,
                friction=0.3,
                )
        rectangle2_fixture=b2FixtureDef(
                shape=b2PolygonShape(box=(.5,2.5)),
                density=1.5,
                friction=0.3,
                )

        self.body1=self.world.CreateBody(
                position=(0, 1),
                fixtures=rectangle1_fixture,
                angle= b2_pi,
                )

        self.body2 = self.world.CreateDynamicBody(
                fixtures=rectangle2_fixture,
                position=(x0[0], x0[1]),
                angle= x0[2] + b2_pi
                )
        self.target = self.world.CreateDynamicBody(
                fixtures=rectangle2_fixture,
                position=(x0[0], x0[1]),
                angle= x0[2]
        )
      


        self.joint=self.world.CreateRevoluteJoint(
                bodyA=self.body1,
                bodyB=self.body2,
                localAnchorB=(0,2.5),
                localAnchorA=(0,-2.5),
                enableMotor = True,
                maxMotorTorque=400,
                enableLimit = False
                )

        self.target_joint = self.world.CreateRevoluteJoint(
                bodyA=self.body1,
                bodyB=self.target,
                localAnchorB=(0,2.5),
                localAnchorA=(0,-2.5),
                enableMotor = False,
                maxMotorTorque=400,
                enableLimit = False,
                )
        pos = self.target.GetWorldPoint((0,2.5))
        self.target.angle = target[0] + b2_pi
        new_pos = self.target.GetWorldPoint((0,2.5))
        self.target.position += pos - new_pos
        self.target.active = False
    def Step(self, settings, action):
        self.joint.motorSpeed = action[0]

        super(ArmWorld, self).Step(settings)

    def reset_world(self):
        self.world.ClearForces()
        self.joint.motorSpeed = 0
        self.body1.linearVelocity= (0, 0)
        self.body1.angularVelocity= 0
        self.body2.linearVelocity = (0, 0)
        self.body2.angularVelocity = 0
        self.body1.position = (0, 1)
        self.body1.angle = b2_pi
        self.body2.position = (self.x0[0], self.x0[1])
        self.body2.angle = self.x0[2] + b2_pi


    def get_state(self):
        state = {POSITION : np.array(self.body2.position), JOINT_ANGLES: np.array([self.joint.angle%(2*b2_pi)]), JOINT_VELOCITIES: np.array([self.joint.speed])}

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
     main(ArmWorld)
