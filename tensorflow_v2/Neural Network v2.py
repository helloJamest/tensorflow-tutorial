import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers,Model

num_features = 28*28
num_classes = 10

learning_rate = 0.01
batch_size = 256
train_steps = 1000
display_steps = train_steps//10

n_hidden_1 = 128
n_hidden_2 = 128


(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train,X_test = np.array(X_train),np.array(X_test)
X_train,X_test = X_train.reshape([-1,28*28]), X_test.reshape([-1,28*28])
X_train,X_test = X_train/255,X_test/255

train_data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

class NeuralNet(Model):
    def __init__(self):
        super(NeuralNet,self).__init__()
        self.fc1 = layers.Dense(n_hidden_1,activation=tf.nn.relu)
        self.fc2 = layers.Dense(n_hidden_2,activation=tf.nn.relu)
        self.out = layers.Dense(num_classes,activation=tf.nn.softmax)

    def call(self,x,istraining=False):
        x = self.fc1(x)
        x = self.fc2(x)
        if not istraining:
            x = self.out(x)
        return x
neuralNet = NeuralNet()

# Cross-Entropy Loss
def cross_entropy(y_pred,y_true):
    y_true = tf.cast(y_true,tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    return tf.reduce_mean(loss)

#Accuracy Metric
def accuracy(y_pred,y_true):
    correction_prediction = tf.equal(tf.argmax(y_pred,1),tf.cast(y_true,tf.int64))
    return tf.reduce_mean(tf.cast(correction_prediction,tf.float32))

optimizer = tf.optimizers.SGD(learning_rate=learning_rate)
def run_optimizer(x,y):
    with tf.GradientTape() as g:
        pred = neuralNet(x,istraining=True)
        loss = cross_entropy(pred,y)
    trainable_variables = neuralNet.trainable_variables
    gradients = g.gradient(loss,trainable_variables)
    optimizer.apply_gradients(zip(gradients,trainable_variables))

for step,(batch_x,batch_y) in enumerate(train_data.take(train_steps),1):
    run_optimizer(batch_x,batch_y)
    if step%display_steps == 0:
        pred = neuralNet(batch_x, is_training=True)
        loss = cross_entropy(pred,batch_y)
        acc = accuracy(pred,batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))


# Test model on validation set.
pred = neuralNet(X_test, is_training=False)
print("Test Accuracy: %f" % accuracy(pred, y_test))

import matplotlib.pyplot as plt
plt.imshow(X_test.reshape((-1,28,28)[0]))
plt.show()


pred = np.argmax(pred.numpy()[0])
print('the pred num is',pred)




