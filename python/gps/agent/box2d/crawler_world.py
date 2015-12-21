
from framework import *
from math import sqrt
from gps.proto.gps_pb2 import *
import numpy as np

# Original inspired by a contribution by roman_m
# Dimensions scooped from APE (http://www.cove.org/ape/index.htm)
class Crawler (Framework):
    def __init__(self):
        super(Crawler, self).__init__()

        ground = self.world.CreateStaticBody(
                shapes=[ 
                        b2EdgeShape(vertices=[(-50,0),(50,0)]),
                        b2EdgeShape(vertices=[(-50,0),(-50,10)]),
                        b2EdgeShape(vertices=[(50,0),(50,10)]),
                    ]
                ) 
        
        self.ground_level = 1.01499

        boxFixture=b2FixtureDef(
                shape=b2PolygonShape(box=(0.5,0.5)),
                density=1,
                friction=0.3)

        square_fixture=b2FixtureDef(
                shape=b2PolygonShape(box=(2.0,1)),
                density=1,
                friction=0.3,
                groupIndex=-1)

        self.square = self.world.CreateDynamicBody(
                fixtures=square_fixture,
                position=((0, self.ground_level)))

        circle=b2FixtureDef(
                shape=b2CircleShape(radius=0.5),
                density=1,
                friction=0.3)
        self.body1=self.world.CreateDynamicBody(
                position=(5, 0.1),
                fixtures=boxFixture
                )
        self.joint=self.world.CreateRevoluteJoint(
                bodyA=self.body1,
                bodyB=self.square,
                anchor=((self.square.position[0] + self.body1.position[0])/2, 5), 
                enableMotor = True,
                maxMotorTorque=400,
                )
        print(self.square.angle)

    def Step(self, settings, action):
        self.body.motorSpeed = action[0]

        super(PointMassWorld, self).Step(settings)

    def reset_world(self):
        self.world.ClearForces()
        self.joint.motorSpeed = 0
        self.square.linearVelocity= (0, 0)
        self.square.angularVelocity= 0
        self.body1.linearVelocity = (0, 0)
        self.body1.angularVelocity = 0
        self.body1.position = (5, 0.1)
        self.body1.angle = 0
        self.square.position = (0, self.ground_level)
        self.square.angle = 0


    def get_state(self):
        state = {POSITION : np.array(self.square.position), LINEAR_VELOCITY : np.array(self.square.linearVelocity)}

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
