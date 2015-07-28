import cPickle
import matplotlib.pyplot as plt
import numpy as np
import scipy.io

def main():
    mat = scipy.io.loadmat('/home/justin/RobotRL/test/onlinecont.mat')
    with open('a_passive_dyn_tv_pure_offline.pkl', 'r') as f:
        oc = cPickle.load(f)
    print 'Loaded data'
    #oc.cost.wu *= 1
    #oc.cost.l1 = 0.1
    #oc.offline_fd = mat['dyn_fd'].transpose(2,0,1)
    #oc.offline_fc = mat['dyn_fc_ref'].T
    #oc.H = 20

    inputs = oc.inputs

    oc.inputs = []
    oc.calculated = []
    for t in range(98):
        X = inputs[t]['x']
        empmu = inputs[t]['empmu']
        empsig = inputs[t]['empsig']
        prevx = inputs[t]['prevx']
        prevu = inputs[t]['prevu']
        tt = inputs[t]['t']
        lgpol = oc.act_pol(X, empmu, empsig, prevx, prevu, tt)

    with open('plot.pkl', 'w') as f:
        cPickle.dump(oc, f)

main()
