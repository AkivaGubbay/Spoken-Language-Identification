'''
A Bidirectional Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np


'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''
# Imports:
import librosa
import os
import numpy as np

# Parameters:
NUM_OF_COEFF = 10   # 10
TIME_S = 30     # 30
TIME_F = 110    # 100
current_batch = 0
# all_mfcc_vectors = []
# all_audio_files = []
# num_of_audio_in_language = 100  # 100
# validation = int(0.15 * num_of_audio_in_language)
# ///////////////


# learning_rate = 0.00001  # without dropout: 0.000001  # with dropout: 0.0000001  # default learning rate of 0.001(for Adam).
# hm_epochs = 600     # without dropout: 600
# n_classes = 4
# batch_size = 1
# chunk_size = (TIME_F - TIME_S)
# n_chunks = NUM_OF_COEFF
# rnn_size = 128  # make this bigger
# up_to_subfile = []

n_steps = NUM_OF_COEFF
n_input = (TIME_F - TIME_S)
n_classes = 4
n_hidden = 128
learning_rate = 0.001
training_iters = 100000
# batch_size = 128
display_step = 10


def getMfccs(audio_file_name):
    wave, sr = librosa.load(audio_file_name, mono=True)
    mfcc = librosa.feature.mfcc(wave, sr)
    try:
        mfcc = mfcc[0:NUM_OF_COEFF, TIME_S: TIME_F]
    except:
        print('audio file: ',audio_file_name, ' was not long enuogh.')
        return None
    return mfcc
'''
def build_up_to_subfile():
    global  up_to_subfile
    up_to_subfile = [True]
    for i in range(1, n_classes):
        up_to_subfile.append(False)

build_up_to_subfile()
'''




def next_batch(directory):
    languages = os.listdir(directory)
    all_mfccs = []
    all_one_hot_vectors = []
    amount_of_languages = len(languages)
    count = 0
    for (i, language) in enumerate(languages):
        one_hot_vec = [0] * amount_of_languages
        one_hot_vec[i] = 1
        lang_dir = directory + '/' + language
        for audio_file in os.listdir(lang_dir):
            audio_file_path = lang_dir + '/' + audio_file
            mfccs = getMfccs(audio_file_path)
            if mfccs is None:
                continue
            all_mfccs.append(mfccs)
            all_one_hot_vectors.append(one_hot_vec)
            count += 1
    # print('all_mfccs:', all_mfccs)
    # print('all_one_hot_vectors:', all_one_hot_vectors)
    all_mfccs_numpy = np.zeros(shape=(count, n_steps * n_input))
    for i in range(0, count):
        '''
        for j in range(0, n_steps * n_input):
            all_mfccs_numpy[i][j] = all_mfccs[i][j]
        '''
        for j in range(0, n_steps * n_input):
            try:
                all_mfccs_numpy[i][j] = all_mfccs[i][j/n_input][j%n_input]
            except:
                print('j:', j)
                print('j/n_input:', j/n_input)
                print('j % n_input:', j % n_input)
    all_one_hot_vectors_numpy = np.zeros(shape=(count, amount_of_languages))
    for i in range(0, count):
        for j in range(0, amount_of_languages):
            all_one_hot_vectors_numpy[i][j] = all_one_hot_vectors[i][j]
    return all_mfccs_numpy, all_one_hot_vectors_numpy






'''
# Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 28 # MNIST data input (img shape: 28*28)
n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
n_classes = 10 # MNIST total classes (0-9 digits)
'''



# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([2*n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def RNN(x, weights, biases):

    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
    # Reshape to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(0, n_steps, x)

    # Define lstm cells with tensorflow
    # Forward direction cell
    lstm_fw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    # Backward direction cell
    lstm_bw_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    try:
        outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                              dtype=tf.float32)
    except Exception: # Old TensorFlow version only returns outputs not states
        outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                        dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

train_directory = r'/media/akiva/Seagate Backup Plus Drive/voxforge/train'
print('building batches..')
batch_x, batch_y = next_batch(train_directory)
print('first batch_x: ', batch_x)
print('first batch_y: ', batch_y)
print('got the batches!')
batch_size = len(batch_x)

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # Reshape data to get 28 seq of 28 elements
        # print('batch_x: ', batch_x)
        batch_x = batch_x.reshape(batch_size, n_steps, n_input)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")


    test_directory = r'/media/akiva/Seagate Backup Plus Drive/voxforge/test'
    batch_x, batch_y = next_batch(test_directory)
    batch_x = batch_x.reshape((-1, n_steps, n_input))
    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={x: batch_x, y: batch_y}))
