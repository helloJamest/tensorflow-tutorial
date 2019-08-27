import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data',one_hot=True)

training_steps = 1000
display_steps = training_steps//10
summary_steps = training_steps//10
learning_rate = 0.01
batch_size = 128
drop_out = 0.6

n_hidden1 = 256
n_hidden2 = 128

num_input = 28*28
num_classes = 10

log_dir = './log'

keep_prob = tf.placeholder(tf.float32,name='keep_prob')
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32,[None,num_input],name='X_input')
    y = tf.placeholder(tf.float32,[None,num_classes],name='y_input')


with tf.name_scope('input_reshape'):
    img_shaped_input = tf.reshape(X,[-1,28,28,1])
    tf.summary.image(name='input',tensor=img_shaped_input,max_outputs=10)


def weight_variable(shape):
    w = tf.Variable(tf.random_normal(shape,stddev=0.1))
    return w

def bias_variable(shape):
    b = tf.Variable(tf.constant(0.1,shape=shape))
    return b

def variable_summary(var):
    with tf.name_scope('summaries'):
        # 计算参数的均值，并使用tf.summary.scaler记录
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean',mean)

        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev',stddev)
        tf.summary.scalar('max',tf.reduce_max(var))
        tf.summary.scalar('min',tf.reduce_min(var))

        # 用直方图记录参数的分布
        tf.summary.histogram('histogram',var)



def nn_layer(x,input_shape,output_shape,layer_name,activation=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable([input_shape,output_shape])
            variable_summary(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_shape])
            variable_summary(biases)
        with tf.name_scope('linear_compute'):
            z  = tf.add(tf.matmul(x,weights),biases)
            # variable_summary(z)
            tf.summary.histogram('z',z)
        a = activation(z)
        tf.summary.histogram('a',a)
    return a

h1 = nn_layer(X,num_input,n_hidden1,'layer1')


with tf.name_scope('drop_out'):
    tf.summary.scalar('drop_out_keep_probability',keep_prob)
    d1 = tf.nn.dropout(h1,keep_prob=keep_prob)

logits = nn_layer(d1,n_hidden1,num_classes,'layer2',activation=tf.identity)
predictions = tf.nn.softmax(logits)

with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=y))

tf.summary.scalar('loss',loss)

with tf.name_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(predictions,1),tf.argmax(y,1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        tf.summary.scalar('accuracy',accuracy)

merged = tf.summary.merge_all()


def feed_dict(train):
    if train:
        xs,ys = mnist.train.next_batch(batch_size)
        k = drop_out
    else:
        xs,ys = mnist.test.images,mnist.test.labels
        k = 1
    return {X:xs,y:ys,keep_prob:k}




init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(log_dir + '/test')

    for i in range(1,training_steps+1):
        # if i % (summary_steps*10) == 0:  # 记录测试集的summary与accuracy
        if i % 10 == 0:  # 记录测试集的summary与accuracy
            summary,acc = sess.run([merged,accuracy],feed_dict=feed_dict(train=False))
            test_writer.add_summary(summary,i)
            print('The Step: %s ; The Accuracy: %s'%(i,acc))
        else:
            if i % 100 == 99:   # 记录训练集的summary
                run_options= tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = sess.run([merged, train_op], feed_dict=feed_dict(train=True),options=run_options,run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata,'step%03d'%i)
                train_writer.add_summary(summary,i)
                print('Adding run metadata for:',i)

            else:
                summary,_ = sess.run([merged,train_op],feed_dict=feed_dict(train=True))
                train_writer.add_summary(summary,i)
    train_writer.close()
    test_writer.close()








# y.eval()
# tensorboard --logdir=

# tensorboard --logdir=~/PycharmProjects/tensorflow_v1/log/train