import cPickle
import matplotlib.pyplot as plt
import numpy as np

def main():
	with open('plot.pkl', 'r') as f:
		oc = cPickle.load(f)

	idx_to_plot = slice(0,7)
	#idx_to_plot = slice(21,30)
	#idx_to_plot = slice(7,14)
	#idx_to_plot = slice(14,21)
	actual_info = oc.inputs
	actual_x = np.array([inp['x'][idx_to_plot] for inp in actual_info])

	fwd_info = oc.fwd_hist
	start_time = 50
	fwd_mu = np.array([mu[idx_to_plot] for mu in oc.fwd_hist[start_time]['trajmu']])
	diff = fwd_mu - actual_x[start_time:start_time+oc.H]
	plt.plot(np.arange(start_time,start_time+oc.H), diff)
	plt.show()

	u_offline = np.array([ oc.offline_K[t].dot(oc.inputs[t]['x'])+oc.offline_k[t]  for t in range(1,98)  ])
	plt.plot(np.arange(97), u_offline)
	plt.show()
	u_online = np.array([ oc.calculated[t]['u'] for t in range(97) ])
	plt.plot(np.arange(2,97), u_online[2:])
	print [oc.calculated[t]['k'][0] for t in range(97)]
	plt.show()


main()
