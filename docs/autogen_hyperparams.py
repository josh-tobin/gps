
from gps.agent.config import AGENT, AGENT_BOX2D, AGENT_ROS, AGENT_MUJOCO
from gps.algorithm.config import ALG, ALG_BADMM
from gps.algorithm.traj_opt.config import TRAJ_OPT_LQR
from gps.algorithm.policy.config import INIT_LG, POLICY_PRIOR, POLICY_PRIOR_GMM
# TODO - change COST_TORQUE to COST_ACTION
from gps.algorithm.cost.config import COST_FK, COST_STATE, COST_SUM, COST_TORQUE
from gps.algorithm.dynamics.config import DYN_PRIOR_GMM
from gps.algorithm.policy_opt.config import POLICY_OPT_CAFFE

header = "Experiment configuration"
description = "This page contains all of the settings that are exposed to \
               change via the experiment hyperparams file. See the \
               corresponding config file for a more detailed explanation \
               of each variable."

f = open('hyperparams.md','w')
f.write(header + '\n' + '===\n' + description + '\n*****\n')

f.write('### Algorithm and Optimization\n')

f.write('** Algorithm **\n')
for key in ALG.keys():
  f.write('* ' + key + '\n')

f.write('** Algorithm BADMM **\n')
for key in ALG_BADMM.keys():
  f.write('* ' + key + '\n')

f.write('### Traj Opt LQR\n')
for key in TRAJ_OPT_LQR.keys():
  f.write('* ' + key + '\n')

f.write('### Caffe Policy Optimization\n')
for key in POLICY_OPT_CAFFE.keys():
  f.write('* ' + key + '\n')

f.write('### Policy Prior & GMM\n')
for key in POLICY_PRIOR.keys():
  f.write('* ' + key + '\n')
for key in POLICY_PRIOR_GMM.keys():
  f.write('* ' + key + '\n')

f.write('### Dynamics\n')

f.write('### Dynamics GMM Prior\n')
for key in DYN_PRIOR_GMM.keys():
  f.write('* ' + key + '\n')

f.write('### Cost Function\n')

f.write('### State cost\n')
for key in COST_STATE.keys():
  f.write('* ' + key + '\n')

f.write('### Forward kinematics cost\n')
for key in COST_FK.keys():
  f.write('* ' + key + '\n')

f.write('### Action cost\n')
for key in COST_TORQUE.keys():
  f.write('* ' + key + '\n')

f.write('### Sum of costs\n')
for key in COST_SUM.keys():
  f.write('* ' + key + '\n')

f.write('### Initialization\n')

f.write('### Initial Trajectory Distribution\n')
for key in INIT_LG.keys():
  f.write('* ' + key + '\n')

f.write('### Agent Interfaces\n')

f.write('### Agent superclass\n')
for key in AGENT.keys():
  f.write('* ' + key + '\n')

f.write('### Box2D agent\n')
for key in AGENT_BOX2D.keys():
  f.write('* ' + key + '\n')

f.write('### Mujoco agent\n')
for key in AGENT_MUJOCO.keys():
  f.write('* ' + key + '\n')

f.write('### ROS agent\n')
for key in AGENT_ROS.keys():
  f.write('* ' + key + '\n')

f.close()
