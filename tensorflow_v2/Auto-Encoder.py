import tensorflow as tf
import numpy as np

num_features = 28*28
learning_rate = 0.01
training_steps=500
display_steps = training_steps//10
batch_sizes = 64

n_hidden_1=128
n_hidden_2=64


from tensorflow.keras.datasets import mnist
(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train,X_test =np.array(X_train,np.float32) ,np.array(X_test,np.float32)
X_train,X_test = X_train.reshape([-1,num_features]), X_test.reshape([-1,num_features])
X_train,X_test = X_train/255,X_test/255

train_data = tf.data.Dataset.from_tensor_slices((X_train,y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_sizes).prefetch(1)

random_normal = tf.initializers.RandomNormal()
weights = {
    'encode_h1':tf.Variable(random_normal([num_features,n_hidden_1])),
    'encode_h2':tf.Variable(random_normal([n_hidden_1,n_hidden_2])),
    'decode_h1':tf.Variable(random_normal([n_hidden_2,n_hidden_1])),
    'decode_h2':tf.Variable(random_normal([n_hidden_1,num_features]))
}
biases = {
    'encode_b1': tf.Variable(random_normal([n_hidden_1])),
    'encode_b2': tf.Variable(random_normal([n_hidden_2])),
    'decode_b1': tf.Variable(random_normal([n_hidden_1])),
    'decode_b2': tf.Variable(random_normal([num_features]))
}

def encode(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['encode_h1']),biases['encode_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['encode_h2']),biases['encode_b2']))
    return layer_2

def decode(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x,weights['decode_h1']),biases['decode_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1,weights['decode_h2']),biases['decode_b2']))
    return layer_2

def mean_square(y_true,y_pred):
    return tf.reduce_mean(tf.pow(y_true-y_pred,2))

optimizer = tf.optimizers.Adam(learning_rate=learning_rate)

def run_optimizer(x):
    with tf.GradientTape() as g:
        reconstruce_img = decode(encode(x))
        loss = mean_square(x,reconstruce_img)
    training_variables = list(weights.values())+list(biases.values())
    gradients = g.gradient(loss,training_variables)
    optimizer.apply_gradients(zip(gradients,training_variables))
    return loss

for step,(batch_x,_) in enumerate(train_data.take(training_steps+1)):
    loss = run_optimizer(batch_x)
    if step % display_steps == 0:
        print("step: %i, loss: %f" % (step, loss))



test_data = tf.data.Dataset.from_tensor_slices((X_test,y_test))
test_data = test_data.repeat().batch(batch_sizes).prefetch(1)
import matplotlib.pyplot as plt
# Encode and decode images from test set and visualize their reconstruction.
n = 4
canvas_orig = np.empty((28 * n, 28 * n))
canvas_recon = np.empty((28 * n, 28 * n))
for i, (batch_x, _) in enumerate(test_data.take(n)):
    # Encode and decode the digit image.
    reconstructed_images = decode(encode(batch_x))
    # Display original images.
    for j in range(n):
        # Draw the generated digits.
        img = batch_x[j].numpy().reshape([28, 28])
        canvas_orig[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = img
    # Display reconstructed images.
    for j in range(n):
        # Draw the generated digits.
        reconstr_img = reconstructed_images[j].numpy().reshape([28, 28])
        canvas_recon[i * 28:(i + 1) * 28, j * 28:(j + 1) * 28] = reconstr_img

print("Original Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_orig, origin="upper", cmap="gray")
plt.show()

print("Reconstructed Images")
plt.figure(figsize=(n, n))
plt.imshow(canvas_recon, origin="upper", cmap="gray")
plt.show()








