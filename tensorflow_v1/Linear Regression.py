import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

rng = np.random


learning_rate = 0.01
training_steps = 1000
display_steps = training_steps//10

train_X = np.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
                         7.042,10.791,5.313,7.997,5.654,9.27,3.1])
train_Y = np.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                         2.827,3.465,1.65,2.904,2.42,2.94,1.3])
n_samples = train_X.shape[0]

X,Y = tf.placeholder('float'),tf.placeholder('float')

W = tf.Variable(rng.rand(),name='weight')
b = tf.Variable(rng.rand(),name='bias')

pred = tf.add(tf.multiply(X,W),b)

cost = tf.reduce_mean(tf.pow(pred-Y,2))/(2*n_samples)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_steps):
        for x,y in zip(train_X,train_Y):
            sess.run(optimizer,feed_dict={X:x,Y:y})
        if (epoch+1) % display_steps == 0:
            c = sess.run(cost,feed_dict={X:x,Y:y})
            w_var,b_var = sess.run(W),sess.run(b)
            print('epoch:%i,cost:%f,w:%f,b:%f'%(epoch+1,c,w_var,b_var))
    print('finished')

    c = sess.run(cost,feed_dict={X:x,Y:y})
    w_var,b_var = sess.run(W),sess.run(b)
    print('epoch:%i,cost:%f,w:%f,b:%f'%(epoch+1,c,w_var,b_var))

    plt.plot(train_X,train_Y,'ro',label='Origin data')
    plt.plot(train_X,w_var*train_X+b_var,label='Fitted line')
    plt.legend()
    plt.plot()
    plt.show()







