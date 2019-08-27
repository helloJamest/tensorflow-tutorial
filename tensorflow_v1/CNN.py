import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir','/tmp/data/','Directory for storing data')

num_classes = 10
num_input = 28*28
batch_size = 128
learning_rate = 0.01
training_steps = 1000
display_steps = training_steps//10

mnist = input_data.read_data_sets(FLAGS.data_dir,one_hot=True)

X = tf.placeholder(tf.float32,[None,num_input])
y = tf.placeholder(tf.float32,[None,num_classes])
keep_prob = tf.placeholder(tf.float32)

def weight_variable(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape):
    return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(x,W):
    """
    tf.nn.conv2d功能：给定4维的input和filter，计算出一个2维的卷积结果
    前几个参数分别是input, filter, strides, padding, use_cudnn_on_gpu, ...
    input   的格式要求为一个张量，[batch, in_height, in_width, in_channels],批次数，图像高度，图像宽度，通道数
    filter  的格式为[filter_height, filter_width, in_channels, out_channels]，滤波器高度，宽度，输入通道数，输出通道数
    strides 一个长为4的list. 表示每次卷积以后在input中滑动的距离
    padding 有SAME和VALID两种选项，表示是否要保留不完全卷积的部分。如果是SAME，则保留
    use_cudnn_on_gpu 是否使用cudnn加速。默认是True
    """
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def maxpool(x):
    """
    tf.nn.max_pool 进行最大值池化操作,而avg_pool 则进行平均值池化操作
    几个参数分别是：value, ksize, strides, padding,
    value:  一个4D张量，格式为[batch, height, width, channels]，与conv2d中input格式一样
    ksize:  长为4的list,表示池化窗口的尺寸
    strides: 窗口的滑动值，与conv2d中的一样
    padding: 与conv2d中用法一样。
    """
    return tf.nn.max_pool2d(x,[1,2,2,1],strides=[1,2,2,1],padding='SAME')


def conv_net(x,keep_prob):
    """
    # 第一层
    # 卷积核(filter)的尺寸是5*5, 通道数为1，输出通道为32，即feature map 数目为32
    # 又因为strides=[1,1,1,1] 所以单个通道的输出尺寸应该跟输入图像一样。即总的卷积输出应该为?*28*28*32
    # 也就是单个通道输出为28*28，共有32个通道,共有?个批次
    # 在池化阶段，ksize=[1,2,2,1] 那么卷积结果经过池化以后的结果，其尺寸应该是？*14*14*32
    """
    x = tf.reshape(x,[-1,28,28,1])
    W_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.elu(conv2d(x,W_conv1)+b_conv1)
    h_plool1 = maxpool(h_conv1)

    W_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.elu(conv2d(h_plool1,W_conv2)+b_conv2)
    h_plool2 = maxpool(h_conv2)

    W_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])
    h_plool2_flatten = tf.reshape(h_plool2,[-1,7*7*64])
    h_fc1 = tf.nn.elu(tf.add(tf.matmul(h_plool2_flatten,W_fc1),b_fc1))
    h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob)

    W_fc2 = weight_variable([1024,num_classes])
    b_fc2 = bias_variable([num_classes])
    # y_conv = tf.nn.softmax(tf.matmul(h_fc1_dropout,W_fc2+b_fc2))
    return tf.matmul(h_fc1_dropout,W_fc2+b_fc2)

    # return y_conv


y_conv = conv_net(X,keep_prob)

# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y*tf.log(y_conv+1e-10),reduction_indices=[1]))   # 损失函数，交叉熵
# optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(cross_entropy)
#
# correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))



logits = conv_net(X,keep_prob)
prediction= tf.nn.softmax(logits)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=y))
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


init = tf.global_variables_initializer()

sess = tf.InteractiveSession()
sess.run(init)
for i in range(1,training_steps+1):
    batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
    sess.run(train_op,feed_dict={X:batch_x,y:batch_y,keep_prob:0.6})
    if i % display_steps == 0 or i==1:
        loss,acc = sess.run([cross_entropy,accuracy],feed_dict={X:batch_x,y:batch_y,keep_prob:1.0})
        print('Step:'+str(i)+'; Loss:'+str(loss)+'; Acc:'+str(acc))

print('Test Acc:',sess.run(accuracy,feed_dict={X:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}))



