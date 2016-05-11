import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import numpy as np
import mjcpy
from gps.agent.mjc.agent_mjc import AgentMuJoCo
import imp
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('position', type=float, nargs='+')
    args = parser.parse_args()
    tgt = np.array(args.position) if len(args.position) == 14 else \
            np.concatenate([np.array(args.position), np.zeros(7)])
    hyperparams_file = 'experiments/pr2_mjc_test/hyperparams.py'
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    hyperparams.config['agent']['x0'] = tgt
    agent = AgentMuJoCo(hyperparams.config['agent'])
    #print agent._world[0].get_data()['xpos']
    print "Joint angles: " + str(agent.x0[0][:7])
    print "EE Position: " + str(agent.x0[0][14:17])
if __name__ == '__main__':
    main()
