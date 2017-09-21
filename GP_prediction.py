import tensorflow as tf
import numpy as np
from edward.models import MultivariateNormalTriL
import matplotlib.pyplot as pl

def tf_rbf(a,b,var,lengthscale):
	sqdist = tf.reshape(tf.reduce_sum(tf.pow(a,2),1),[-1,1]) + tf.reduce_sum(tf.pow(b,2),1) - 2*tf.matmul(a, tf.transpose(b))
	return var*tf.exp(-.5 * (1/lengthscale) * sqdist)

n = 50			#number of test points (x)
n_samples = 5	#number of samples from the GP prior
Xtest = np.linspace(-5, 5, n).reshape([n,1])
Xtest = Xtest.astype(np.float32, copy=False)

# Noisy training data
ntrain = 10
Xtrain = np.array([-4, -3, -2.8, -2, -1, 2, 2.5, 3.2, 3.7, 4]).reshape(ntrain,1)
ytrain = np.sin(Xtrain) + np.random.normal(0,0.1,[ntrain,1])

X = tf.placeholder(tf.float32, [ntrain, 1])
Y = tf.placeholder(tf.float32, [ntrain, 1])
X_star = tf.placeholder(tf.float32, [n, 1])

#Hyperparameters 
var = tf.Variable(1.0, trainable=True)
lengthscale = tf.Variable(1.0, trainable=True)
noise = tf.Variable(1.0, trainable=True)

#Sub-matrices
K = tf_rbf(X,X,var,lengthscale) + noise*tf.eye(ntrain)
K_star = tf_rbf(X,X_star,var,lengthscale)
K_star_star = tf_rbf(X_star,X_star,var,lengthscale)

#Predictive Mean Vector
L_inv = tf.matrix_inverse(tf.cholesky(K))
alpha = tf.matmul(tf.matmul(tf.transpose(L_inv),L_inv),Y)
mu = tf.matmul(tf.transpose(K_star),alpha)
mu = tf.reshape(mu,[-1])
#Predictive Covariance Matrix
sigma = K_star_star - tf.matmul(tf.matmul(tf.matmul(tf.transpose(K_star),tf.transpose(L_inv)),L_inv),K_star)

#Posterior Gaussian Distribution
posterior = MultivariateNormalTriL(loc=mu, scale_tril=sigma)
samples = posterior.sample(n_samples)

#Negative Log Likelihood
nll = tf.matrix_determinant(K) + tf.matmul(tf.matmul(tf.transpose(Y),tf.matrix_inverse(K)),Y)
loss = nll
opt = tf.train.AdamOptimizer(0.1).minimize(nll)

#Optimize Hyperparameters
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(1000):
        print(sess.run([var,lengthscale, noise, loss], feed_dict={X: Xtrain, Y: ytrain, X_star: Xtest}))
        sess.run(opt,feed_dict={X: Xtrain, Y: ytrain, X_star: Xtest})

	s = sess.run(samples,feed_dict={X: Xtrain, Y: ytrain, X_star: Xtest})
	s = np.transpose(s)

	mean = sess.run(mu,feed_dict={X: Xtrain, Y: ytrain, X_star: Xtest})
	covariance = sess.run(sigma,feed_dict={X: Xtrain, Y: ytrain, X_star: Xtest})

print mean.shape

stdv = np.sqrt(np.diag(covariance))
print stdv.shape

Xtest = np.reshape(Xtest,[-1])
trp = pl.plot(Xtrain, ytrain, 'ro',label='Training Points')
pl.axis([-5, 5, -5, 5])
pred = pl.plot(Xtest,mean,lw=3,label='Predicted Mean')
fun = pl.plot(Xtest,np.sin(Xtest),label='True function')
pl.fill_between(Xtest, mean-2*stdv, mean+2*stdv,facecolor='silver')
pl.legend(loc='upper left')
pl.plot(Xtest, s, lw=0.5)
pl.title(str(n_samples)+' samples from the GP posterior')
pl.show()
