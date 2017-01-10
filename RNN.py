import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
# mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
from Parameters import *
from PrePorcessing import *
from time import time


x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')
keep_prob = tf.placeholder(tf.float32)


def recurrent_neural_network(x):

    # ======= Fully connected layers ===========

    layer1 = {'weights': tf.Variable(tf.random_normal([rnn_size, 80])),
              'biases': tf.Variable(tf.constant(0.1, shape=[80]))}

    layer2 = {'weights': tf.Variable(tf.random_normal([80, 60])),
              'biases': tf.Variable(tf.constant(0.1, shape=[60]))}

    layer3 = {'weights': tf.Variable(tf.random_normal([60, 30])),
              'biases': tf.Variable(tf.constant(0.1, shape=[30]))}

    layer4 = {'weights': tf.Variable(tf.random_normal([30, 10])),
              'biases': tf.Variable(tf.constant(0.1, shape=[10]))}

    # ==========================================

    layer = {'weights': tf.Variable(tf.random_normal([80, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}  # rnn_size

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    '''
    lstm_cell = tf.nn.rnn_cell.LSTMCell(rnn_size, state_is_tuple=True)
    lstm_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_cell, input_keep_prob=input_dropout, output_keep_prob=output_dropout)
    lstm_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 9, state_is_tuple=True)    # rnn_size  # 3                                 # What does this number mean??
    '''

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True)

    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output1 = tf.matmul(outputs[-1], layer1['weights']) + layer1['biases']
    output1 = tf.nn.relu(output1)
    output1 = tf.nn.dropout(output1, keep_prob)
    '''
    output2 = tf.matmul(output1, layer2['weights']) + layer2['biases']
    output2 = tf.nn.relu(output2)

    output3 = tf.matmul(output2, layer3['weights']) + layer3['biases']
    output3 = tf.nn.relu(output3)

    output4 = tf.matmul(output3, layer4['weights']) + layer4['biases']
    output4 = tf.nn.relu(output4)
    '''

    output = tf.matmul(output1, layer['weights']) + layer['biases']     # outputs[-1]

    return output


def train_neural_network(x):
    global current_batch, up_to_subfile, validation, num_of_audio_in_language, learning_rate

    # Calculating mfccs vectors!!!
    directory = r'/media/akiva/Seagate Backup Plus Drive/voxforge/parent'       # The 'r' is to prevent white spaces.
    audio_to_mfccs(directory, 0, num_of_audio_in_language)

    prediction = recurrent_neural_network(x)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)      # Amos said to try gradient decent. AdamOptimizer

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        first = True    # To reduce learning-rate once.
        second = True   # To reduce learning-rate twice.

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int((4 * num_of_audio_in_language) / batch_size)):
                # print('Ok\tepoch:', epoch, 'iter:', _)

                # epoch_x is of passed as shape: (1 X coeff X time). 1 is actually the batch_size.
                epoch_x, epoch_y = next_batch(num_of_audio_in_language)
                # print('epoch_x:', epoch_x, '\n')

                # Case: passed the length of data set.
                if epoch_x is None and epoch_y is None:
                    print('My ERROR: passed the length of data set.(RRN module).')
                    break

                try:
                    epoch_x = epoch_x.reshape(batch_size, n_chunks, chunk_size)
                except:
                    print('reshape problem.\tepoch:', epoch, 'iter:', _)
                    continue

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y, keep_prob: 0.9})  # , keep_prob: 0.7
                epoch_loss += c

            print('*************** Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss, '**********')

            if epoch_loss < 350 and first is True:
                learning_rate /= 100
                first = False
            elif epoch_loss < 80 and second is True:
                learning_rate /= 100
                second = False
            # Stop training stage early:
            elif epoch_loss < 0.01:
                break


        # Evaluate model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

        # ===============================Test & Accuracy================================================================

        print('\n====================Test stage===================================\n')

        # Create mfcc vectors of validation set:
        # from: amount of audio files in EACH language.
        # to: from + the size validation(per language).
        audio_to_mfccs(directory, num_of_audio_in_language, num_of_audio_in_language + validation)     # directory, 0, num_of_audio_in_language
        total_accuracy = 0
        total_amount_of_audio = 4 * validation  # KILL THIS
        for i in range(4 * validation):     # int((4 * num_of_audio_in_language) / batch_size)
            # print('test file:', i)
            epoch_x, label = next_batch(validation)     # num_of_audio_in_language


            try:
                currect_test = accuracy.eval({x: epoch_x.reshape((-1, n_chunks, chunk_size)), y: label, keep_prob: 1.0})    # , keep_prob: 1.0
            except:
                print('Accuracy reshape problem.')
                total_amount_of_audio -= 1
                continue

            total_accuracy += currect_test
            # print('Accuracy:',
            #      currect_test)  # accuracy.eval({x: epoch_x.reshape((-1, n_chunks, chunk_size)), y: label}))  # accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))

        print('Total Accuracy: ', (total_accuracy / total_amount_of_audio) * 100, '%')   # (4 * num_of_audio_in_language))

        '''
        print('========================Testing on training data=============================================')
        audio_to_mfccs(directory, 0, num_of_audio_in_language)
        total_accuracy = 0
        for i in range(int((4 * num_of_audio_in_language) / batch_size)):
            epoch_x, label = next_batch(num_of_audio_in_language)
            try:
                currect_test = accuracy.eval({x: epoch_x.reshape((-1, n_chunks, chunk_size)), y: label})
            except:
                print('Accuracy reshape problem.')
                continue

            total_accuracy += currect_test

        print('Total Accuracy: ', (total_accuracy / (4 * num_of_audio_in_language)) * 100, '%')

        # =============================================================================================================
        '''


s = time()
train_neural_network(x)
e = time()

print('\n\ntime: ', (e - s)/60, 'min')