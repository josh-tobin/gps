import scipy.io
import numpy as np

class GaussianProcess(object):
	def __init__(self, matfile):
		params = scipy.io.loadmat(matfile)
		self.alpha_1 = params['alpha_1'][0]  # RBF variance
		self.beta = params['beta'][0]  # RBF inverse width
		self.alpha_2 = params['alpha_2'][0]  # Bias variance
		self.Kinvy = params['Kinvy']
		self.X = params['X']
		self.Xstd = params['Xstd'][0]
		self.Xmean = params['Xmean'][0]
		self.ymean = params['ymean']
		self.ystd = params['ystd'][0]
		self.kern = 'rbf'

		self.din = self.X.shape[1]
		self.dout = self.Kinvy.shape[1]
		self.N = self.X.shape[0]

	def eval(self, pt):
		if self.kern == 'rbf':
			normpt = (pt-self.Xmean)/self.Xstd
			diff = normpt-self.X
			diff = np.sum(diff*diff, axis=1)
			kernel_term = self.alpha_1*np.exp(-(0.5*self.beta)*diff)  # RBF
		else:
			ip = self.X.dot(pt)
			kernel_term = self.alpha_1*ip # Linear

		k = kernel_term +self.alpha_2
		result = (k.dot(self.Kinvy)*self.ystd) + self.ymean
		return result[0]

	def kerneval(self, pt):
		if self.kern == 'rbf':
			normpt = (pt-self.Xmean)/self.Xstd
			diff = normpt-self.X
			diff = np.sum(diff*diff, axis=1)
			kernel_term = self.alpha_1*np.exp(-(0.5*self.beta)*diff)  # RBF
			return kernel_term
		raise NotImplementedError()

	def linearize(self, pt):
		if self.kern == 'rbf':
			# Forward pass
			normpt = (pt-self.Xmean)/self.Xstd
			diff = normpt-self.X
			diff = np.sum(diff*diff, axis=1)
			kernel_term = self.alpha_1*np.exp(-(0.5*self.beta)*diff)
			k = kernel_term +self.alpha_2
			result = (k.dot(self.Kinvy)*self.ystd) + self.ymean

			# Backward pass
			dAdk = self.Kinvy.T*np.expand_dims(self.ystd, axis=1) # (dY x N)
			negbeta = -0.5*self.beta
			#dkdd_diag = self.alpha_1 * (negbeta) * np.exp(negbeta*diff)  # (N,)
			dkdd_diag = (negbeta) * kernel_term # (N,)
			#dkdd = np.diag(dkdd_diag) # NxN
			dddx = 2*(normpt-self.X)/np.expand_dims(self.Xstd, axis=0)  # (N x dX)
			dkdx = np.expand_dims(dkdd_diag, axis=1)*dddx
			dAdx = dAdk.dot(dkdx)  # (dY x dX)

		else:
			raise NotImplementedError()  # Need to incorporate normalization
			ip = self.X.dot(pt)
			kernel_term = self.alpha_1*ip # Linear
			k = kernel_term +self.alpha_2
			result = k.dot(self.Kinvy) + self.ymean

			# Backward pass
			#dAdk = self.Kinvy.T  # (dY x N)
			#dkdd_diag = self.alpha_1 * (negbeta) * np.exp(negbeta*diff)  # (N,)
			#dkdd_diag = np.ones(self.N)*self.alpha_1 # (N,)
			#dkdd = np.diag(dkdd_diag) # NxN
			#dipdx = self.X  # (N x dX)
			#dkdx = np.expand_dims(dkdd_diag, axis=1)*dipdx
			#dAdx = dAdk.dot(dkdx)  # (dY x dX)
			dAdx = self.alpha_1*self.Kinvy.T.dot(self.X)


		# Taylor expansion
		F = dAdx
		f = -dAdx.dot(pt) + result
		return F, f

def finite_differences(gp, pt, eps=1e-5):
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
	gp = GaussianProcess('/home/justin/data/gp_pr2_4000.mat')
	x = (gp.X[0,:]*gp.Xstd)+gp.Xmean
	print gp.eval(x)
	F, f = gp.linearize(x)
	grad = finite_differences(gp, x, eps=1e-5)
	
	max_diff = np.max(np.abs(F-grad)) 
	grad_check = max_diff < 2e-5
	if grad_check:
		print "Gradient good!"
	else:
		print "Gradient bad!"
		print "Max diff:", max_diff
	import pdb; pdb.set_trace()

