import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=True)

learning_rate = 0.1
training_steps = 100
display_steps = training_steps//10
batch_size =64

X = tf.placeholder(tf.float32,[None,28*28])
y = tf.placeholder(tf.float32,[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

pred = tf.nn.softmax(tf.add(tf.matmul(X,W),b))

cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred)))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_steps):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict={X:batch_x,y:batch_y})
            avg_cost += c/total_batch
        if (epoch+1) %display_steps == 0:
            print('The epoch:%i,The cost:%9f'%(epoch+1,avg_cost))
    print('Finished')
    correct_prediction = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print('acc:',sess.run(accuracy,feed_dict={X: mnist.test.images[:3000], y: mnist.test.labels[:3000]}))

