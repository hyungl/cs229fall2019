import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from preprocess import *

X_train, X_test, y_train, y_test = get_train_test()
print(X_train.shape)
X_train = X_train.reshape(X_train.shape[0], 660)
X_test = X_test.reshape(X_test.shape[0], 660)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
oneHot = OneHotEncoder(categories='auto')
oneHot.fit(y_train)
y = oneHot.transform(y_train).toarray()
oneHot.fit(y_test)
y_t = oneHot.transform(y_test).toarray()
# number of features
num_features = 660
# number of target labels
num_labels = 35
# learning rate (alpha)
learning_rate = 0.01
# batch size
batch_size = 128
# number of epochs
num_steps = 100001

# input data
train_dataset = X_train
print(train_dataset.shape)
train_labels = y
print(train_labels.shape)
test_dataset = X_test
print(test_dataset.shape)
test_labels = y_t
print(test_labels.shape)


# initialize a tensorflow graph
graph = tf.Graph()

with graph.as_default():
    """ 
    defining all the nodes 
    """

    # Inputs
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    weights = tf.Variable(tf.truncated_normal([num_features, num_labels]))
    biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(tf_train_dataset, weights) + biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=tf_train_labels, logits=logits))

    # Optimizer.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    train_prediction = tf.nn.softmax(logits)
    test_prediction = tf.nn.softmax(tf.matmul(tf_test_dataset, weights) + biases)


# utility function to calculate accuracy
def accuracy(predictions, labels):
    correctly_predicted = np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
    accu = (100.0 * correctly_predicted) / predictions.shape[0]
    return accu


with tf.Session(graph=graph) as session:
    # initialize weights and biases
    tf.global_variables_initializer().run()
    print("Initialized")

    for step in range(num_steps):
        # pick a randomized offset
        offset = np.random.randint(0, train_labels.shape[0] - batch_size - 1)

        # Generate a minibatch.
        batch_data = train_dataset[offset:(offset + batch_size), :]
        batch_labels = train_labels[offset:(offset + batch_size), :]

        # Prepare the feed dict
        feed_dict = {tf_train_dataset: batch_data,
                     tf_train_labels: batch_labels}

        # run one step of computation
        _, l, predictions = session.run([optimizer, loss, train_prediction],
                                        feed_dict=feed_dict)

        if (step % 500 == 0):
            print("Minibatch loss at step {0}: {1}".format(step, l))
            print("Minibatch accuracy: {:.1f}%".format(
                accuracy(predictions, batch_labels)))

    print("\nTest accuracy: {:.1f}%".format(
        accuracy(test_prediction.eval(), test_labels)))

