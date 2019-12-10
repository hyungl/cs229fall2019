import tensorflow.compat.v1 as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from preprocess import *
import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import os

#https://www.geeksforgeeks.org/softmax-regression-using-tensorflow/

def plot_confusion_matrix(cm, classes,normalize=True,title=None,cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    classes = sorted(classes)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax = ax)
    ax.set(title=title,ylabel='True label',xlabel='Predicted label')

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "/Users/hyung.lee/cs229fall2019/speech_commands_tenlabels/"
num_classes = int(sys.argv[2]) if len(sys.argv) > 2 else 10
n_mfcc = int(sys.argv[3]) if len(sys.argv) > 3 else 20

x_train, x_validate, x_test, y_train, y_validate, y_test = get_train_test_valid(path=DATA_PATH)
x_train = x_train.reshape(x_train.shape[0], n_mfcc)
x_validate = x_validate.reshape(x_validate.shape[0], n_mfcc)
x_test = x_test.reshape(x_test.shape[0], n_mfcc)
y_train = y_train.reshape(y_train.shape[0], 1)
y_validate = y_validate.reshape(y_validate.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)
oneHot = OneHotEncoder(categories='auto')
oneHot.fit(y_train)
y = oneHot.transform(y_train).toarray()
oneHot.fit(y_validate)
y_validate = oneHot.transform(y_validate).toarray()
oneHot.fit(y_test)
y_test = oneHot.transform(y_test).toarray()
# number of features
num_features = n_mfcc
# number of target labels
num_labels = int(sys.argv[2])
# learning rate (alpha)
learning_rate = 0.05
# batch size
batch_size = 128
# number of epochs
num_steps = 35001

# input data
train_dataset = x_train
train_labels = y
valid_dataset = x_validate
valid_labels = y_validate
test_dataset = x_test
test_labels = y_test

# initialize a tensorflow graph
graph = tf.Graph()

with graph.as_default():
    """ 
    defining all the nodes 
    """

    # Inputs
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, num_features))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
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
    valid_prediction = tf.nn.softmax(tf.matmul(tf_valid_dataset, weights) + biases)
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
            print("Validation accuracy: {:.1f}%".format(
                accuracy(valid_prediction.eval(), valid_labels)))

    print("\nTest accuracy: {:.1f}%".format(
        accuracy(test_prediction.eval(), test_labels)))

cnf_matrix = confusion_matrix(test_labels, test_prediction.eval())
class_names = ['one','two','three','four','five','six','seven','eight','nine','zero']
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True, title='Normalized Confusion Matrix')
plot_name = 'ConfusionMatrix_Normalized_' + 'Test_' + 'Set_Multinomial.pdf'
plt.savefig(os.path.join('.', plot_name))
