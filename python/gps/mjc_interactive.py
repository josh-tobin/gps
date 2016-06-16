import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import numpy as np
import mjcpy
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.utility.general_utils import find_objects_in_model
import imp
import argparse
import xml.etree.ElementTree as ElementTree

def _parse(elt):
    #print elt.body
    if isinstance(elt, list):
        print "Is list of length %d"%len(elt)
        for e in elt:
            _parse(e)
    else:
        print "Is elt"
        _parse(elt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('position', type=float, nargs='+')
    args = parser.parse_args()
    tgt = np.array(args.position) if len(args.position) == 14 else \
            np.concatenate([np.array(args.position), np.zeros(7)])
    hyperparams_file = 'experiments/crl_swingup_ff/hyperparams.py'
    
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    f =  hyperparams.config['agent']['filename']
    hyperparams.config['agent']['x0'] = tgt
    agent = AgentMuJoCo(hyperparams.config['agent'])
    #print agent._world[0].get_data()['xpos']
    #print [method for method in dir(agent._model) if callable(getattr(agent._model, method))]
    print agent._model[0]['nq']
    print agent._model[0]['nv']

    #for name in agent._model[0]['names']:
    #    print mjcpy.id2name(name)
    #print agent._model[0]['names']
    #print agent._model[0]['name_geomadr']
    #print agent._world[0].get_data().keys()
    #new_xfrc[30,2] = -9.81
    #print agent._world[0].set_data({'xfrc_applied': new_xfrc})
    
    #print agent._world[0].get_data().keys()
    objects = find_objects_in_model(f)
    print "objects"
    print objects
    
    agent._set_gravity(0)
    #new_xfrc = agent._world[0].get_data()['xfrc_applied']
    #for object_id in objects.values():
    #    new_xfrc[object_id, 2] = -9.81 * 
    U = np.zeros(7)
    X = np.append(tgt, np.zeros(13)) 
    X[7] = 0.95
    X[8] = 0.2
    X[9] = 0.6
    for t in range(1000):
        agent._world[0].plot(X)
        for _ in range(50):
            X, _ = agent._world[0].step(X, U)
    
if __name__ == '__main__':
    main()
