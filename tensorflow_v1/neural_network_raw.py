import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=True)

learning_rate = 0.01
training_steps = 5000
display_steps = training_steps//10

batch_size = 128

n_hidden_1 = 128
n_hidden_2 = 128

num_features = 28*28
num_classes = 10

X = tf.placeholder(tf.float32,[None,num_features])
y = tf.placeholder(tf.float32,[None,num_classes])

weights = {
    'w1':tf.Variable(tf.random_normal([num_features,n_hidden_1])),
    'w2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'w3':tf.Variable(tf.random_normal([n_hidden_2,num_classes]))
}

biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'b3':tf.Variable(tf.random_normal([num_classes]))
}

def neural_network(x):
    # h1 = tf.nn.relu(tf.add(tf.matmul(x,weights['w1']),biases['b1']))
    # h2 = tf.nn.relu(tf.add(tf.matmul(h1,weights['w2']),biases['b2']))
    h1 = tf.add(tf.matmul(x,weights['w1']),biases['b1'])
    h2 = tf.add(tf.matmul(h1,weights['w2']),biases['b2'])
    out = tf.add(tf.matmul(h2,weights['w3']),biases['b3'])
    return out

logits = neural_network(X)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss_op)

correct_op = tf.equal(tf.argmax(logits,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_op,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(1,training_steps+1):
        batch_x,batch_y = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={X:batch_x,y:batch_y})
        if epoch % display_steps == 0 or epoch == 1:
            loss,acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x,y:batch_y})
            print('step:'+str(epoch)+' ;Mini batch Loss:{:6f}'.format(loss)+' ;Accuracy:{:2f}'.format(acc))
    print('finished')
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: mnist.test.images,
                                        y: mnist.test.labels}))









