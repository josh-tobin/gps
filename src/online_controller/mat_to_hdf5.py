import h5py
import scipy.io
import train_dyn_net
import sys
import os

#fname = sys.argv[1]
#prefix, ext = os.path.splitext(fname)
infiles = sys.argv[1].split(',')
outfile = h5py.File(sys.argv[2])
data, lbl, _, _ = train_dyn_net.get_data(infiles, infiles, remove_ft=True)

outfile['data'] = data
outfile['label'] = lbl

outfile.flush()
