import numpy as np
import tensorflow as tf
import edward as ed
import matplotlib
import matplotlib.pyplot as pl

#Radial Basis Function
def tf_rbf(a,b,var,lengthscale):
	sqdist = tf.reshape(tf.reduce_sum(tf.pow(a,2),1),[-1,1]) + tf.reduce_sum(tf.pow(b,2),1) - 2*tf.matmul(a, tf.transpose(b))
	return var*tf.exp(-.5 * (1/lengthscale) * sqdist)

#Load Oil Flow Data
data = np.loadtxt('oil_data/DataTrn.txt')
labels = np.loadtxt('oil_data/DataTrnLbls.txt')

#Subset of examples
ntrain = 100
latentDim = 2

data = data[:ntrain,:]
labels = labels[:ntrain,:]
lbls = np.zeros(ntrain)

#Labels
i = 0
for row in labels:
	if (row == [1.,0.,0.]).all():
		lbls[i] = 0
	if (row == [0.,1.,0.]).all():
		lbls[i] = 1
	if (row == [0.,0.,1.]).all():
		lbls[i] = 2
	i = i + 1

colors = ['red','green','blue']

init = tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)	#Random Gaussian Initialization on the latent space 
X = tf.get_variable("latent", shape=[ntrain,latentDim], initializer=init, trainable=True)
Y = tf.placeholder(tf.float32, [ntrain, 12])

#Hyperparameters 
var = tf.Variable(1.0, trainable=True)				#RBF Variance
lengthscale = tf.Variable(1.0, trainable=True)		#RBF Lenghtscale
noise = tf.Variable(1.0, trainable=True)			#Model's noise

#Covariance Matrix
K = tf_rbf(X,X,var,lengthscale)+ noise*tf.eye(ntrain)

#Negative Log evidence
nle = tf.matrix_determinant(K) + tf.trace(tf.matmul(tf.matmul(tf.transpose(Y),tf.matrix_inverse(K)),Y))		
loss = nle
opt = tf.train.AdamOptimizer(0.01).minimize(loss)

#Optimize Hyperparameters and X
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
    	print 'Iteration no ' + str(i)
        print(sess.run([var,lengthscale, noise, loss], feed_dict={Y: data}))
        sess.run(opt,feed_dict={Y: data})
    
	latent_pos = sess.run(X)

print latent_pos.shape

#Plot Learned Latent Space
pl.scatter(latent_pos[:,0], latent_pos[:,1], c=lbls, cmap=matplotlib.colors.ListedColormap(colors), s=80)
pl.show()
