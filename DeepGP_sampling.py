import tensorflow as tf
import edward as ed
import numpy as np
from edward.models import MultivariateNormalTriL
from edward.util import rbf
import matplotlib.pyplot as pl
from edward.models import Normal

def is_pos_def(A):
	def f1(): return True
	def f2(): return False
	e,_ = tf.self_adjoint_eig(A)
	r = tf.cond(tf.reduce_all(tf.greater(e,0)), f1, f2)
	return r

def tf_jitchol(A,maxtries=5.):
	diagA = tf.diag_part(A);
	jitter = tf.reduce_mean(diagA) * 1e-7
	num_tries = tf.constant(1,dtype=tf.float32)
	def body(num_tries,A,jitter,maxtries):
		jitter *= tf.pow(10.,num_tries)
		r = tf.cond(is_pos_def(A), lambda: tf.cholesky(A), lambda: tf.add(A,tf.multiply(tf.eye(tf.shape(A)[0]),jitter)))
		tf.add(num_tries, 1)
		return num_tries,r,jitter,maxtries
	condition = lambda num_tries,A,jitter,maxtries: tf.logical_and(tf.less(num_tries, maxtries),tf.logical_not(is_pos_def(A)))
	result = tf.while_loop(condition, body, [num_tries,A,jitter,maxtries], parallel_iterations=1)	
	return  result[1]
		
n = 100			#number of test points (x)
n_samples = 3	#number of samples from the GP prior
Xtest = np.linspace(-5, 5, n).reshape([n,1])
Xtest = Xtest.astype(np.float32, copy=False)

X = tf.placeholder(tf.float32, [n, 1])

covar = tf_jitchol(rbf(X),10.);
GP1 = MultivariateNormalTriL(loc=tf.zeros(n), scale_tril=covar)
samp = GP1.sample(1)
samp = tf.transpose(samp)
covar2 = tf_jitchol(rbf(samp),10.);
GP2 = MultivariateNormalTriL(loc=tf.zeros(n), scale_tril=covar2)

samples = GP2.sample(n_samples)

sess = tf.Session()
s = sess.run(samples,feed_dict={X: Xtest})
s = np.transpose(s)
print s.shape

pl.plot(Xtest, s)
pl.axis([-5, 5, -10, 10])
pl.title(str(n_samples)+' samples from the GP prior')
pl.show()
