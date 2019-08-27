import tensorflow as tf
import numpy as np
from tensorflow.contrib.factorization import KMeans
import os

os.environ['CUDA_VISIBLE_DEVICES']=''

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data',one_hot=True)

full_data_x = mnist.train.images

num_steps = 50
batch_size = 1024
k = 25
num_classes = 10
num_features = 28*28

X = tf.placeholder(tf.float32,[None,num_features])
y = tf.placeholder(tf.float32,[None,num_classes])

kmeans = KMeans(inputs=X,num_clusters=k,distance_metric='cosine',use_mini_batch=True)

# Build KMeans graph
all_scores, cluster_idx, scores, cluster_centers_initialized,init_op, training_op = kmeans.training_graph()

cluster_idx = cluster_idx[0]
avg_distance = tf.reduce_mean(scores)

# Initialize the variables (i.e. assign their default value)
init_vars = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init_vars, feed_dict={X: full_data_x})
sess.run(init_op, feed_dict={X: full_data_x})

# Training
for i in range(1, num_steps + 1):
    _, d, idx = sess.run([training_op, avg_distance, cluster_idx],
                         feed_dict={X: full_data_x})
    if i % 10 == 0 or i == 1:
        print("Step %i, Avg Distance: %f" % (i, d))

counts = np.zeros(shape=(k, num_classes))
for i in range(len(idx)):
    counts[idx[i]] += mnist.train.labels[i]
# Assign the most frequent label to the centroid
labels_map = [np.argmax(c) for c in counts]
labels_map = tf.convert_to_tensor(labels_map)

# Evaluation ops
# Lookup: centroid_id -> label
cluster_label = tf.nn.embedding_lookup(labels_map, cluster_idx)
# Compute accuracy
correct_prediction = tf.equal(cluster_label, tf.cast(tf.argmax(y, 1), tf.int32))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Test Model
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, y: test_y}))





