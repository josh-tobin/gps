#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# C++ version Copyright (c) 2006-2007 Erin Catto http://www.box2d.org
# Python version Copyright (c) 2010 kne / sirkne at gmail dot com
# 
# This software is provided 'as-is', without any express or implied
# warranty.  In no event will the authors be held liable for any damages
# arising from the use of this software.
# Permission is granted to anyone to use this software for any purpose,
# including commercial applications, and to alter it and redistribute it
# freely, subject to the following restrictions:
# 1. The origin of this software must not be misrepresented; you must not
# claim that you wrote the original software. If you use this software
# in a product, an acknowledgment in the product documentation would be
# appreciated but is not required.
# 2. Altered source versions must be plainly marked as such, and must not be
# misrepresented as being the original software.
# 3. This notice may not be removed or altered from any source distribution.

from framework import *
from math import sqrt
from gps.proto.gps_pb2 import *
import numpy as np

class PointMassWorld (Framework):
    name="PointMass"
    def __init__(self, position=(0,2), angle=b2_pi, linearVelocity=(0,0), angularVelocity=0, target = (12, 24)):
        super(PointMassWorld, self).__init__()
        self.world.gravity = (0.0, 0.0)
        self.initial_position = position
        self.initial_angle = angle
        self.initial_linearVelocity = linearVelocity
        self.initial_angularVelocity = angularVelocity

        # The boundaries
        ground = self.world.CreateBody(position=(0, 20))
        ground.CreateEdgeChain(
                            [ (-20,-20),
                              (-20, 20),
                              ( 20, 20),
                              ( 20,-20),
                              (-20,-20) ]
                            )


        xf1 = b2Transform()
        xf1.angle = 0.3524 * b2_pi
        xf1.position = b2Mul(xf1.R, (1.0, 0.0))

        xf2 = b2Transform()
        xf2.angle = -0.3524 * b2_pi
        xf2.position = b2Mul(xf2.R, (-1.0, 0.0))
        vertices=[xf1*(-1,0), xf1*(1,0), xf1*(0,.5)]
        self.body = self.world.CreateDynamicBody(
                    position=self.initial_position, 
                    angle=self.initial_angle,
		    linearVelocity=self.initial_linearVelocity,
		    angularVelocity=self.initial_angularVelocity,
                    angularDamping=5,
                    linearDamping=0.1,
                    shapes=[b2PolygonShape(vertices=[xf1*(-1,0), xf1*(1,0), xf1*(0,.5)]),
                            b2PolygonShape(vertices=[xf2*(-1,0), xf2*(1,0), xf2*(0,.5)]) ],
                    shapeFixture=b2FixtureDef(density=2.0),
                )
        # self.target = self.world.CreateStaticBody(
        #             position = target,
        #             shapes = [b2PolygonShape(vertices=[xf1*(-1,0), xf1*(1,0), xf1*(0,.5)]),
        #                     b2PolygonShape(vertices=[xf2*(-1,0), xf2*(1,0), xf2*(0,.5)]) ],
                # )
    def Step(self, settings, action):
        """Called upon every step. """
        self.body.ApplyTorque(action, True)
        print(action)

        super(PointMassWorld, self).Step(settings)


    def reset_world(self):
        self.world.ClearForces()
        self.body.position = self.initial_position
        self.body.angle = self.initial_angle
        self.body.angularVelocity = self.initial_angularVelocity
        self.body.linearVelocity = self.initial_linearVelocity

    def get_state(self):
        state = {POSITION : np.array(self.body.position), ANGLE : [self.body.angle], LINEAR_VELOCITY : np.array(self.body.linearVelocity), ANGULAR_VELOCITY : [self.body.angularVelocity]}
	return state


if __name__=="__main__":
    # main(PointMassWorld)
    world = PointMassWorld(position=(0,2), target=(0,2))
    world.run()
    i = 1
    while True:
        print(world.get_state()[POSITION])
