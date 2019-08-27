import tensorflow as tf
import numpy as np
from tensorflow.keras import layers,Model

num_classes = 10

learning_rate = 0.01
batch_sizes = 128
train_steps = 1000
display_steps = train_steps//10

conv1_filters = 32
conv2_filters = 64
fc1_units = 1024


from tensorflow.keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()

X_train, X_test = np.array(X_train,np.float32), np.array(X_test,np.float32)
X_train, X_test = X_train/255, X_test/255

train_data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_sizes).prefetch(1)

class ConvNet(Model):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conV1 = layers.Conv2D(conv1_filters,kernel_size=5,activation = tf.nn.relu)
        self.maxpool1 = layers.MaxPool2D(2,strides=2)

        self.conV2 = layers.Conv2D(conv2_filters,kernel_size=3,activation=tf.nn.relu)
        self.maxpool2 = layers.MaxPool2D(2,strides=2)

        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(fc1_units)
        self.dropout = layers.Dropout(rate=0.7)
        self.out = layers.Dense(num_classes)


    def call(self,x,is_training=False):
        x = tf.reshape(x,[-1,28,28,1])
        x = self.conV1(x)
        x = self.maxpool1(x)
        x = self.conV2(x)
        x = self.maxpool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.out(x)
        if not is_training:
            x = tf.nn.softmax(x)
        return x

conv_net = ConvNet()


def cross_entropy(x,y):
    y = tf.cast(y,tf.int64)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=x)
    return tf.reduce_mean(loss)

def accuracy(y_pred,y_true):
    correction_prediction = tf.equal(tf.argmax(y_pred,1),tf.cast(y_true,tf.int64))
    return tf.reduce_mean(tf.cast(correction_prediction,tf.float64))


optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

def run_optimizer(x,y):
    with tf.GradientTape() as g:
        pred = conv_net(x,is_training=True)
        loss = cross_entropy(pred,y)
    trainable_variables = conv_net.trainable_variables
    gradients = g.gradient(loss,trainable_variables)
    optimizer.apply_gradients(zip(gradients,trainable_variables))


for step,(batch_x,batch_y) in enumerate(train_data.take(train_steps),1):
    run_optimizer(batch_x,batch_y)
    if step % display_steps == 0:
        pred = conv_net(batch_x)
        loss = cross_entropy(pred,batch_y)
        acc = accuracy(pred,batch_y)
        print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))

pred  = conv_net(X_test)
acc = accuracy(pred,y_test)
print('the test acc: %f'%acc)


import matplotlib.pyplot as plt
plt.imshow(X_test[0])
plt.show()

pred = np.argmax(pred.numpy()[0])
print('the num pred is :',pred)






