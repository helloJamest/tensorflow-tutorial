import tensorflow as tf
import numpy as  np
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets('/tmp/data',one_hot=True)


learning_rate = 0.01
training_steps = 1000
display_steps = training_steps//10
batch_size = 128

num_input = 28
timesteps = 28
num_hidden = 120
num_classes = 10

X = tf.placeholder('float',[None,timesteps,num_input])
y = tf.placeholder('float',[None,num_classes])

weights ={
    'out':tf.Variable(tf.random_normal([num_hidden,num_classes]))
}

biases = {
    'out':tf.Variable(tf.random_normal([num_classes]))
}

def RNN(x,weights,biases):
    x = tf.unstack(x,timesteps,1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden,forget_bias=1.0)
    outputs,states = rnn.static_rnn(lstm_cell,x,dtype=tf.float32)
    return tf.matmul(outputs[-1],weights['out'])+biases['out']

logits = RNN(X,weights,biases)
prediction =tf.nn.softmax(logits)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(1,training_steps+1):
        batch_x,batch_y = mnist.train.next_batch(batch_size)

        batch_x = batch_x.reshape([batch_size,timesteps,num_input])
        sess.run(train_op,feed_dict={X:batch_x,y:batch_y})

        if step == 1 or step % display_steps == 0:
            loss,acc = sess.run([loss_op,accuracy],feed_dict={X:batch_x,y:batch_y})
            print('Step:'+str(step)+'; The Loss:'+str(loss)+'; The acc:'+str(acc))
    print('Training Finished')

    test_len=128
    test_data = mnist.test.images[:test_len].reshape([-1,timesteps,num_input])
    test_label = mnist.test.labels[:test_len]
    print('Test accuracy:%f'%(sess.run(accuracy,feed_dict={X:test_data,y:test_label})))






















