
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np


# Imports:
import librosa
import os
import numpy as np
from time import time


# Parameters:
NUM_OF_COEFF = 10   # 10
TIME_S = 30     # 30
TIME_F = 110    # 100
current_batch = 0
train_directory = r'/media/akiva/Seagate Backup Plus Drive/voxforge/two_languages_parent/about_1500_training_files/train'
test_directory = r'/media/akiva/Seagate Backup Plus Drive/voxforge/two_languages_parent/about_1500_training_files/test'
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
n_classes = 2
n_hidden = 128
learning_rate = 0.001
training_iters = 800000  # 100000    # 4_abount_1500: 800000
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


def next_batch(directory):
    languages = os.listdir(directory)
    print(languages)
    all_mfccs = []
    all_one_hot_vectors = []
    amount_of_languages = len(languages)
    count = 0
    for (i, language) in enumerate(languages):
        print('calculating MFCC for', language, 'language..')
        one_hot_vec = [0] * amount_of_languages
        one_hot_vec[i] = 1
        lang_dir = directory + '/' + language
        for audio_file in os.listdir(lang_dir): # [0:10]:   # limited audio..
            audio_file_path = lang_dir + '/' + audio_file
            mfccs = getMfccs(audio_file_path)

            # Checking if audio file is suitable:
            if mfccs is None:
                continue
            skip_this = False
            for i in range(0, len(mfccs)):   # or NUM_OF_COEFF.
                if len(mfccs[i]) != n_input:
                    skip_this = True
                    break
            if skip_this is True:
                continue

            # Audio file is ok - add it:
            all_mfccs.append(mfccs)
            all_one_hot_vectors.append(one_hot_vec)
            count += 1

    print('number of suitable files: ', count)
    print('converting to numpy..')
    # print('all_mfccs:', all_mfccs)
    # print('all_one_hot_vectors:', all_one_hot_vectors)
    all_mfccs_numpy = np.zeros(shape=(count, n_steps * n_input))
    for i in range(0, count):
        '''
        for j in range(0, n_steps * n_input):
            all_mfccs_numpy[i][j] = all_mfccs[i][j]
        '''
        for j in range(0, n_steps * n_input):
            # print('i:', i)
            # print('j:', j)
            # print('j/n_input:', int(j / n_input))
            # print('j % n_input:', j % n_input)

            # print('\nall_mfccs[i][int(j/n_input)][j % n_input]:', all_mfccs[i][int(j / n_input)][j % n_input])
            # print('all_mfccs_numpy[i][j]:', all_mfccs_numpy[i][j])
            try:
                all_mfccs_numpy[i][j] = all_mfccs[i][int(j/n_input)][j % n_input]
            except:
                print(' np array problem: (', i, ',', int(j/n_input), ',', j % n_input, ')')
                all_mfccs_numpy[i][j] = 0
                continue    # Don't really need this..

    all_one_hot_vectors_numpy = np.zeros(shape=(count, amount_of_languages))
    for i in range(0, count):
        for j in range(0, amount_of_languages):
            all_one_hot_vectors_numpy[i][j] = all_one_hot_vectors[i][j]
    return all_mfccs_numpy, all_one_hot_vectors_numpy


# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
keep_prob = tf.placeholder("float")

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))     # 2*n_hidden
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}




def RNN(x, weights, biases):
    global keep_prob

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

    # Fully connected layer:
    fully_connected_layer = {'weights': tf.Variable(tf.random_normal([2*n_hidden, n_hidden])),
                             'biases': tf.Variable(tf.constant(0.1, shape=[n_hidden]))}    # 2*n_hidden

    z = tf.matmul(outputs[-1], fully_connected_layer['weights']) + fully_connected_layer['biases']
    z = tf.nn.relu(z)
    z = tf.nn.dropout(z, keep_prob)

    z1 = tf.matmul(z, weights['out']) + biases['out']
    return z1

s = time()
pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

print('building batches..')
batch_x, batch_y = next_batch(train_directory)
print('got the batches!')
batch_size = len(batch_x)
print('\n================ Conducting Training =====================')
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    '''
    # save model:
    saver = tf.train.Saver()
    saver.save(sess, 'my-model')
    # load model:
    new_saver = tf.train.import_meta_graph('my-model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./'))
    '''

    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        # Reshape data to get 28 seq of 28 elements
        # print('batch_x: ', batch_x)
        batch_x = batch_x.reshape(batch_size, n_steps, n_input)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.6})  #, keep_prob: 0.9
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
            print("Iter " + str(step*batch_size) + ", Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))

            if acc == 1.0:
                print('Exited early - reached high training accuracy.')
                break

        step += 1
    print("Optimization Finished!")

    print('\n================ Testing Stage ===========================')

    batch_x, batch_y = next_batch(test_directory)
    batch_x = batch_x.reshape((-1, n_steps, n_input))
    print("\nTesting Accuracy:", \
          sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0}))   # , keep_prob: 1.0


e = time()
print('\n\ntime: ', (e - s)/60, 'min')