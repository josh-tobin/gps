import cPickle
import matplotlib.pyplot as plt
import numpy as np
import sys

np.set_printoptions(suppress=True)
def main():
	with open(sys.argv[1], 'r') as f:
		oc = cPickle.load(f)

	idx_to_plot = slice(0,7)
	#idx_to_plot = slice(21,30)
	#idx_to_plot = slice(7,14)
	#idx_to_plot = slice(14,21)
	actual_info = oc[0]
	actual_x = np.array([inp['x'][idx_to_plot] for inp in actual_info])

	fwd_info = oc[1]


	"""
	start_time = 5
	print 'T=',start_time
	fwd_mu = np.array([mu[idx_to_plot] for mu in oc.fwd_hist[start_time]['trajmu']])
	diff = fwd_mu - actual_x[start_time:start_time+oc.H]
	plt.plot(np.arange(start_time,start_time+oc.H), diff)
	plt.show()

	start_time = 50
	print 'T=',start_time
	fwd_mu = np.array([mu[idx_to_plot] for mu in oc.fwd_hist[start_time]['trajmu']])
	diff = fwd_mu - actual_x[start_time:start_time+oc.H]
	plt.plot(np.arange(start_time,start_time+oc.H), diff)
	plt.show()

	start_time = 80
	print 'T=',start_time
	fwd_mu = np.array([mu[idx_to_plot] for mu in oc.fwd_hist[start_time]['trajmu']])
	diff = fwd_mu - actual_x[start_time:start_time+oc.H]
	plt.plot(np.arange(start_time,start_time+oc.H), diff)
	plt.show()

	u_offline = np.array([ oc.offline_K[t].dot(oc.inputs[t]['x'])+oc.offline_k[t]  for t in range(1,98)  ])
	plt.plot(np.arange(97), u_offline)
	plt.show()
	"""
	#u_online = np.array([ oc.calculated[t]['u'] for t in range(97) ])
	#plt.plot(np.arange(2,97), u_online[2:])
	#plt.show()

	dynamics = oc[1]
	F = []
	f = []
	empsig = []
	for t in range(1, len(dynamics)):
		if 'old' not in dynamics[t]:
			print dynamics[t].keys()
			continue
		else:
			print t
		F.append(dynamics[t]['old']['F'][0])
		f.append(dynamics[t]['old']['f'][0])
		empsig.append(dynamics[t]['old']['empsig'])

	import pdb; pdb.set_trace()


main()
