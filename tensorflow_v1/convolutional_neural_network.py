import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=False)

learning_rate = 0.01
training_steps = 200
display_steps = training_steps//10
batch_size = 128

num_input = 28*28
num_classes = 10
dropout = 0.75

#
def conv_net(x_dict,n_classes,dropout,reuse,is_training):
    with tf.variable_scope('Convnet',reuse=reuse):
        x = x_dict['images']
        x = tf.reshape(x,shape=[-1,28,28,1])
        conv1 = tf.layers.conv2d(x,32,5,activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1,2,2)

        conv2 = tf.layers.conv2d(conv1,64,3,activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2,2,2)

        fc1 = tf.contrib.layers.flatten(conv2)

        fc1= tf.layers.dense(fc1,1024)
        fc1 = tf.layers.dropout(fc1,rate=dropout,training=is_training)

        out = tf.layers.dense(fc1,n_classes)

    return out
# Create the neural network
# def conv_net(x_dict, n_classes, dropout, reuse, is_training):
#     # Define a scope for reusing the variables
#     with tf.variable_scope('ConvNet', reuse=reuse):
#         # TF Estimator input is a dict, in case of multiple inputs
#         x = x_dict['images']
#
#         # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
#         # Reshape to match picture format [Height x Width x Channel]
#         # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
#         x = tf.reshape(x, shape=[-1, 28, 28, 1])
#
#         # Convolution Layer with 32 filters and a kernel size of 5
#         conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
#         # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
#         conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
#
#         # Convolution Layer with 64 filters and a kernel size of 3
#         conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
#         # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
#         conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
#
#         # Flatten the data to a 1-D vector for the fully connected layer
#         fc1 = tf.contrib.layers.flatten(conv2)
#
#         # Fully connected layer (in tf contrib folder for now)
#         fc1 = tf.layers.dense(fc1, 1024)
#         # Apply Dropout (if is_training is False, dropout is not applied)
#         fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
#
#         # Output layer, class prediction
#         out = tf.layers.dense(fc1, n_classes)

    # return out

# Define the model function (following TF Estimator Template)
def model_fn(features,labels,mode):
    logits_train = conv_net(features,num_classes,dropout,reuse=False,is_training=True)
    logits_test = conv_net(features,num_classes,dropout,reuse=True,is_training=False)

    pred_classes = tf.argmax(logits_test,axis=1)
    pred_prob = tf.nn.softmax(logits_test)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode,predictions=pred_classes)

    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train,labels=tf.cast(labels,dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op,global_step=tf.train.get_global_step())
    acc_op = tf.metrics.accuracy(labels=labels,predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss_op,
        train_op=train_op,
        eval_metric_ops={'accuracy':acc_op}
    )
    return estim_specs


model = tf.estimator.Estimator(model_fn)
input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'images':mnist.train.images},
    y = mnist.train.labels,
    batch_size=batch_size,
    num_epochs = None,
    shuffle=True
)
model.train(input_fn,steps=training_steps)




input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'images':mnist.test.images},
    y = mnist.test.labels,
    batch_size=batch_size,
    shuffle=True
)
model.evaluate(input_fn)

n_images = 4
test_images = mnist.test.images[:n_images]


input_fn = tf.estimator.inputs.numpy_input_fn(
    x = {'images':test_images},
    shuffle=False
)

preds = list(model.predict(input_fn))
for i in range(n_images):
    plt.imshow(test_images[i].reshape([28,28]),cmap='gray')
    plt.show()
    print('Model prediction',preds[i])
