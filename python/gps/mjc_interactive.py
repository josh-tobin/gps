import sys
sys.path.append('/'.join(str.split(__file__, '/')[:-2]))
import numpy as np
import mjcpy
from gps.agent.mjc.agent_mjc import AgentMuJoCo
from gps.utility.general_utils import find_objects_in_model, find_bodies_in_model
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
    hyperparams_file = 'experiments/crl_box_rnn_1mass/hyperparams.py'
    
    hyperparams = imp.load_source('hyperparams', hyperparams_file)
    f =  hyperparams.config['agent']['filename']
    hyperparams.config['agent']['x0'] = tgt
    agent = AgentMuJoCo(hyperparams.config['agent'])
    #print agent._world[0].get_data()['xpos']
    #print [method for method in dir(agent._model) if callable(getattr(agent._model, method))]
    print agent._world[0].get_data().keys()
    print
    print agent._model[0]['dof_frictional']
    #print agent._world[0].get_data()['efc_force']
    #print agent._world[0].get_data()['qfrc_bias']

    #for name in agent._model[0]['names']:
    #    print mjcpy.id2name(name)
    #print agent._model[0]['names']
    #print agent._model[0]['name_geomadr']
    #print agent._world[0].get_data().keys()
    #new_xfrc[30,2] = -9.81
    #print agent._world[0].set_data({'xfrc_applied': new_xfrc})
    
    #print agent._world[0].get_data().keys()
    #objects = find_objects_in_model(f)
    #print "objects"
    #print objects
    
    arm_links = ['l_shoulder_pan_link', 'l_shoulder_lift_link', 'l_upper_arm_roll_link', 'l_upper_arm_link', 'l_elbow_flex_link', 'l_forearm_roll_link', 'l_forearm_link']
    bodies = find_bodies_in_model(f, arm_links)
    #print 'bodies'
    #print bodies
    
    objects = find_objects_in_model(f)
    print 'objects'
    print objects
    
    #print agent._world[0].get_data()['body_pos'][objects.values()[0],:]
    print agent._model[0]['dof_damping'].shape
    print
#    print agent._world[0].get_data()
    #agent._model[0]['body_pos'][objects.values()[0],:] *= 2
    #agent._model[0]['geom_size'][objects.values()[0]-1,:] *= 2
    #print agent._model[0]['body_pos'][objects.values()[0],:]
    #print agent._model[0]['geom_size'][objects.values()[0]-1,:]
    #agent._world[0].set_model(agent._model[0])
    #print agent._world[0].get_data()['geom_size'][objects.values()[0],0]
    #print agent._model[0]['geom_xpos'][objects.values()[0],:]
    
    #agent._set_gravity(0)
    
    #new_xfrc = agent._world[0].get_data()['xfrc_applied']
    #for object_id in objects.values():
    #    new_xfrc[object_id, 2] = -9.81 * 
    U = np.zeros(7)
    X = tgt
    #X = np.append(tgt, np.zeros(13)) 
    #X[7] = 0.95
    #X[8] = 0.2
    #X[9] = 0.6
    for t in range(1000):
        agent._world[0].plot(X)
        for _ in range(50):
            X, _ = agent._world[0].step(X, U)

if __name__ == '__main__':
    main()
