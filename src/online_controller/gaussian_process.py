import scipy.io
import numpy as np

class GaussianProcess(object):
	def __init__(self, matfile):
		params = scipy.io.loadmat(matfile)
		self.alpha_1 = params['alpha_1'][0]
		self.beta = params['beta'][0]
		self.alpha_2 = params['alpha_2'][0]
		self.Kinvy = params['Kinvy']
		self.X = params['X']
		self.ymean = params['ymean']

		self.din = self.X.shape[1]
		self.dout = self.Kinvy.shape[1]

	def eval(self, pt):
		diff = pt-self.X
		diff = np.sum(diff*diff, axis=1)
		k = self.alpha_1*np.exp(-(0.5*self.beta)*diff)+self.alpha_2
		result = k.dot(self.Kinvy) + self.ymean
		return result[0]

	def linearize(self, pt):
		# Forward pass
		diff = pt-self.X
		diff = np.sum(diff*diff, axis=1)
		k = self.alpha_1*np.exp(-(0.5*self.beta)*diff)+self.alpha_2
		result = k.dot(self.Kinvy) + self.ymean

		# Backward pass
		dAdk = self.Kinvy.T  # (dY x N)
		negbeta = -0.5*self.beta
		dkdd_diag = self.alpha_1 * (negbeta) * np.exp(negbeta*diff)  # (N,)
		#dkdd = np.diag(dkdd_diag) # NxN
		dddx = 2*(pt-self.X)  # (N x dX)
		dkdx = np.expand_dims(dkdd_diag, axis=1)*dddx
		dAdx = dAdk.dot(dkdx)  # (dY x dX)

		# Taylor expansion
		F = dAdx
		f = -dAdx.dot(pt) + result
		return F, f

def gradient_check(gp, pt, eps=1e-5):
	dout = gp.dout
	din = gp.din
	grad = np.zeros((dout, din))
	for idx_i in range(din):
		pt[idx_i] += eps
		out_1 = gp.eval(pt)
		pt[idx_i] -= 2*eps
		out_2 = gp.eval(pt)
		pt[idx_i] += eps

		out_diff = (out_1-out_2)/(2*eps)

		grad[:,idx_i] += out_diff
	return grad

if __name__ == "__main__":
	np.set_printoptions(suppress=True)
	gp = GaussianProcess('/home/justin/data/gp_pr2_2000.mat')
	x = gp.X[0,:]
	print gp.eval(x)
	F, f = gp.linearize(x)
	grad = gradient_check(gp, x)
	
	max_diff = np.max(np.abs(F-grad)) 
	grad_check = max_diff < 1e-5
	if grad_check:
		print "Gradient good!"
	else:
		print "Gradient bad!"
		print "Max diff:", max_diff

