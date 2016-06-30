import sys
sys.path.append('python')

from gps.algorithm.policy.model_predictive_controller import ModelPredictiveController
from gps.algorithm.policy.lin_gauss_policy import LinearGaussianPolicy 
from gps.utility.data_logger import DataLogger
import imp
from timeit import default_timer as timer
BASE_DIR = '/home/jt/gps/'
EXP_DIR = 'experiments/'
DATA_DIR = '/data_files/'
exp_name = 'ilqr_example'
filename = 'policy_cond_0'
hyperparams_ext = '/hyperparams.py'

policy_dir = BASE_DIR + EXP_DIR + exp_name + DATA_DIR + filename
hyperparams_dir = BASE_DIR + EXP_DIR + exp_name + hyperparams_ext
hyperparams = imp.load_source('hyperparams', hyperparams_dir)
algorithm_file = BASE_DIR + EXP_DIR + exp_name + DATA_DIR + 'algorithm_itr_14.pkl'

'''
def main():
    dl = DataLogger()
    policy = LinearGaussianPolicy.load_policy(policy_dir)
    agent = hyperparams.config['agent']['type'](hyperparams.config['agent'])
    hyperparams.config['algorithm'].update({'agent': agent})
    algo = hyperparams.config['algorithm']['type'](hyperparams.config['algorithm'])
    traj_opt = hyperparams.config['algorithm']['traj_opt']['type'](hyperparams.config['algorithm']['traj_opt'])    
    mpc_policy = ModelPredictiveController(policy, algo.cur[0].traj_info, algo, 0)
    print algo.cur[0].traj_info.Cm             
    agent.sample(mpc_policy, 0)
'''

def main():
    dl = DataLogger()
    algorithm = dl.unpickle(algorithm_file)
    traj_opt = algorithm.traj_opt
    agent = hyperparams.config['agent']['type'](hyperparams.config['agent'])
    policy = algorithm.cur[0].traj_distr
    traj_info = algorithm.cur[0].traj_info
    
    pol_sample_lists = dl.unpickle(BASE_DIR + EXP_DIR + exp_name + DATA_DIR + 'traj_sample_itr_14.pkl')
    for n in range(len(pol_sample_lists[0])):
        pol_sample_lists[0][n].agent = agent
    algorithm.cur[0].traj_info.dynamics.agent = agent
    algorithm.cur[0].sample_list = pol_sample_lists[0]
    algorithm._eval_cost(0)
    mpc_policy = ModelPredictiveController(algorithm, 0)

    start = timer() 
    agent.sample(mpc_policy, 0)
    end = timer()
    print mpc_policy.error_history
    print 'Total time: %.2f'%(end-start)

if __name__ == '__main__':
    main()
