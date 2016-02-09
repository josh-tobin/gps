
from framework import *
from math import sqrt
from gps.proto.gps_pb2 import *
import numpy as np

# Original inspired by a contribution by roman_m
# Dimensions scooped from APE (http://www.cove.org/ape/index.htm)
class ArmWorld(Framework):
    def __init__(self, x0=[0, 2, 0, 0], target=[10]):
        super(ArmWorld, self).__init__()

        fixture_length = 3
        self.x0 = x0



        rectangle_fixture=b2FixtureDef(
                shape=b2PolygonShape(box=(.5, fixture_length)),
                density=.5,
                friction=1,
                )
        square_fixture=b2FixtureDef(
                shape=b2PolygonShape(box=(1,1)),
                density=1.5,
                friction=1,
                )
        self.base = self.world.CreateBody(
                position=(0, 7),
                fixtures=square_fixture,
            )

        self.body1=self.world.CreateDynamicBody(
                position=(0, 2),
                fixtures=rectangle_fixture,
                angle= b2_pi,
                )

        self.body2 = self.world.CreateDynamicBody(
                fixtures=rectangle_fixture,
                position=(0, 2),
                angle= b2_pi
                )
        self.target1 = self.world.CreateDynamicBody(
                fixtures=rectangle_fixture,
                position=(0, 0),
                angle = b2_pi
        )
        self.target2 = self.world.CreateDynamicBody(
                fixtures=rectangle_fixture,
                position=(0, 0),
                angle = b2_pi
        )
      


        self.joint1=self.world.CreateRevoluteJoint(
                bodyA=self.base,
                bodyB=self.body1,
                localAnchorA=(0,0),
                localAnchorB=(0,3),
                enableMotor = True,
                maxMotorTorque=400,
                enableLimit = False
                )

        self.joint2=self.world.CreateRevoluteJoint(
                bodyA=self.body1,
                bodyB=self.body2,
                localAnchorB=(0,2.5),
                localAnchorA=(0,-2.5),
                enableMotor = True,
                maxMotorTorque=400,
                enableLimit = False
                )

        self.set_joint_angles(self.body1, self.body2, x0[0], x0[1])
        self.set_joint_angles(self.target1, self.target2, target[0], target[1])
        self.target1.active = False
        self.target2.active = False

        self.joint1.motorSpeed = x0[2]
        self.joint2.motorSpeed = x0[3]

    

    def set_joint_angles(self, body1, body2, angle1, angle2):
        pos = self.base.GetWorldPoint((0,0))
        body1.angle = angle1 + b2_pi
        new_pos = body1.GetWorldPoint((0,3))
        body1.position += pos - new_pos
        body2.angle = angle2 + body1.angle
        pos = body1.GetWorldPoint((0,-2.5))
        new_pos = body2.GetWorldPoint((0,3))
        body2.position += pos - new_pos
    def Step(self, settings, action):
        self.joint1.motorSpeed = action[0]
        self.joint2.motorSpeed = action[1]

        super(ArmWorld, self).Step(settings)

    def reset_world(self):
        self.world.ClearForces()
        self.joint1.motorSpeed = 0
        self.joint2.motorSpeed = 0
        self.body1.linearVelocity= (0, 0)
        self.body1.angularVelocity= 0
        self.body2.linearVelocity = (0, 0)
        self.body2.angularVelocity = 0
        self.set_joint_angles(self.body1, self.body2, self.x0[0], self.x0[1])


    def get_state(self):
        state = {JOINT_ANGLES: np.array([self.joint1.angle%(2*b2_pi), self.joint2.angle%(2*b2_pi)]),
                JOINT_VELOCITIES: np.array([self.joint1.speed, self.joint2.speed])}

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
