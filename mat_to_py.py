import h5py

fname = '/home/justin/RLL_SVN/justinf_branch/baxter_plane_nn_04-22-20_19_itr1.mat'
mat = h5py.File(fname);
exp_state = mat['experiment_state']

print exp_state.keys()
import pdb ; pdb.set_trace()
