
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.proto.gps_pb2 import JOINT_ANGLES, JOINT_VELOCITIES, \
        END_EFFECTOR_POINTS, END_EFFECTOR_POINT_VELOCITIES, ACTION

SENSOR_DIMS = {
    JOINT_ANGLES: 7,
    JOINT_VELOCITIES: 7,
    END_EFFECTOR_POINTS: 3 * EE_POINTS.shape[0],
    END_EFFECTOR_POINT_VELOCITIES: 3 * EE_POINTS.shape[0],
    ACTION: 7,
}

agent = {
    'type': AgentMuJoCo,
    'filename': '../../mjc_models/PR2/pr2_1arm_coarse.xml',
    'x0': np.zeros(14),
    'dt': 0.05,
    'substeps': 1,
    'ee_points_tgt': np.zeros(14),
    'conditions': 1,
    'pos_body_id': np.array([1]),
    'pos_body_offset': np.array([0.0, 0.0, 0.0]),
    'T': 100,
    'sensor_dims': SENSOR_DIMS,
    'state_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
                    END_EFFECTOR_POINT_VELOCITIES],
    'obs_include': [JOINT_ANGLES, JOINT_VELOCITIES, END_EFFECTOR_POINTS,
        END_EFFECTOR_POINT_VELOCITIES],
}
