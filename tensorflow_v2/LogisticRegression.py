import tensorflow as tf
import numpy as np

num_classes = 10
num_features = 28*28

train_steps = 100
display_steps = 100
batch_size = 256
learning_rate = 0.01

from tensorflow.keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()

X_train,X_test = np.array(X_train,np.float32),np.array(X_test,np.float32)
X_train,X_test = X_train.reshape([-1,num_features]),X_test.reshape([-1,num_features])
X_train,X_test = X_train/255,X_test/255

train_data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

W = tf.Variable(tf.ones([num_features,num_classes]),name='weight')
b= tf.Variable(tf.zeros([num_classes]))

# Logistic regression (Wx + b).
def logistic_regress(x):
    return tf.nn.softmax(tf.matmul(x,W)+b)

# Cross-Entropy loss function.
def cross_entropy(y_pred,y_true):
    y_true = tf.one_hot(y_true,depth=num_classes)
    y_pred = tf.clip_by_value(y_pred,1e-9,1)
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

# Accuracy metric.
def accuracy(y_pred,y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred,1),tf.cast(y_true,tf.int64))
    return tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


optimizer = tf.keras.optimizers.SGD(learning_rate = learning_rate)
# Optimization process.
def run_optimizer(x,y):
    with tf.GradientTape() as g:
        y_pred = logistic_regress(x)
        loss = cross_entropy(y_pred,y)
    gradients = g.gradient(loss,[W,b])
    optimizer.apply_gradients(zip(gradients,[W,b]))

for step,(batch_x,batch_y) in enumerate(train_data.take(train_steps),1):
    run_optimizer(batch_x,batch_y)
    if step%display_steps==0:
        y_pred = logistic_regress(batch_x)
        loss = cross_entropy(y_pred,batch_y)
        acc = accuracy(y_pred,batch_y)
        print('step :{0},loss:{1},acc:{2}'.format(step,loss,acc))


pred = logistic_regress(X_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))

import matplotlib.pyplot as plt
test_shape = X_test.shape[0]
plt.imshow(X_test.reshape([-1,28,28])[0])
plt.show()

y_pred = logistic_regress([X_test[0]])
print('the image prediction is :',tf.argmax(y_pred,1).numpy()[0])
#np.argmax(y_pred.numpy())


