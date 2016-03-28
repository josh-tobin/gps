#!/bin/bash
rostopic pub /gps_controller_position_command gps_agent_pkg/PositionCommand '{id: 0, mode: 1, arm: 0, data: [0.1, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0], pd_gains: [3.09, 1.08, 0.393, 0.674, 0.111, 0.152, 0.098]}'
