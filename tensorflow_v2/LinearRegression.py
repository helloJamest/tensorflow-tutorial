import tensorflow as tf
import numpy as np

train_steps = 1000
display_steps = 100
learning_rate = 0.01

X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = X.shape[0]

W = tf.Variable(np.random.rand()*0.01,name='weight')
b = tf.Variable(np.random.rand()*0.01,name='bias')

def linear_regress(x):
    return W * x + b

def mean_square_error(y_pred,y_true):
    return tf.reduce_sum(tf.pow(y_pred-y_true,2))/(2*n_samples)

optimizers = tf.optimizers.SGD(learning_rate=learning_rate)

def run_optimizer():
    with tf.GradientTape() as g:
        pred = linear_regress(x=X)
        loss  = mean_square_error(pred,Y)
    gradients = g.gradient(loss,[W,b])

    optimizers.apply_gradients(zip(gradients,[W,b]))

for step in range(1,train_steps+1):
    run_optimizer()
    if step % display_steps == 0:
        pred = linear_regress(X)
        loss = mean_square_error(pred,Y)
        print('step :{0},loss:{1}'.format(step,loss))



import matplotlib.pyplot as plt

# Graphc plot
plt.plot(X,Y,'ro',label='Original data')
plt.plot(X,np.array(W*X+b),label='Fitted Line')

plt.legend()
plt.show()

