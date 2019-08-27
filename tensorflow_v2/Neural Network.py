import tensorflow as tf
import numpy as np

num_features = 28*28
num_classes = 10

learning_rate = 0.01
train_steps = 300
display_steps = train_steps//10
batch_size = 256

n_hidden_1 = 128
n_hidden_2 = 64

from tensorflow.keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train, X_test = np.array(X_train,np.float32), np.array(X_test,np.float32)
X_train, X_test = X_train.reshape([-1,28*28]), X_test.reshape([-1,28*28])
X_train, X_test = X_train/255, X_test/255

train_data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# A random value generator to initialize weights.
random_normal = tf.initializers.RandomNormal()
weights = {
    'h1':tf.Variable(random_normal([num_features,n_hidden_1])),
    'h2':tf.Variable(random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(random_normal([n_hidden_2,num_classes]))
}
biases = {
    'b1':tf.Variable(tf.zeros([n_hidden_1])),
    'b2':tf.Variable(tf.zeros([n_hidden_2])),
    'out':tf.Variable(tf.zeros([num_classes]))
}


# Create model.
def neuralNet(x):
    h1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['h1']), biases['b1']))
    h2 = tf.nn.sigmoid(tf.add(tf.matmul(h1,weights['h2']), biases['b2']))
    out = tf.nn.softmax(tf.add(tf.matmul(h2,weights['out']), biases['out']))
    return out

#Corss Entropy loss function
def cross_entropy(y_pred,y_true):
    y_true = tf.one_hot(y_true,depth=num_classes)
    y_pred = tf.clip_by_value(y_pred,1e-9,1)
    return tf.reduce_mean(-tf.reduce_sum(y_true*tf.math.log(y_pred)))


def accuracy(y_pred,y_true):
    correction = tf.equal(tf.argmax(y_pred,1),tf.cast(y_true,tf.int64))
    return tf.reduce_mean(tf.cast(correction,tf.float32))


optimizer = tf.optimizers.SGD(learning_rate = learning_rate)

def run_optimizer(x,y):
    with tf.GradientTape() as g:
        pred = neuralNet(x)
        loss = cross_entropy(pred,y)
    train_variable = list(weights.values())+list(biases.values())
    gradients = g.gradient(loss,train_variable)
    optimizer.apply_gradients(zip(gradients,train_variable))

for step,(batch_x,batch_y) in enumerate(train_data.take(train_steps),1):
    run_optimizer(batch_x,batch_y)
    if step % display_steps == 0:
        pred = neuralNet(batch_x)
        loss = cross_entropy(pred,batch_y)
        acc = accuracy(pred,batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))


pred = neuralNet(X_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

import matplotlib.pyplot as plt
plt.imshow(X_test.reshape([-1,28,28])[0])
plt.show()

y_pred = neuralNet([X_test[0]])
print('the image prediction is :',tf.argmax(y_pred,1).numpy()[0])
#np.argmax(y_pred.numpy())