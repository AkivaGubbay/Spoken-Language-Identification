import tensorflow as tf
import numpy as np
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.ops import rnn, rnn_cell
# mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)
from Parameters import *
from PrePorcessing import *

'''
hm_epochs = 50   # A lot more..
n_classes = 10  # 4
batch_size = 128
chunk_size = 28
n_chunks = 28
rnn_size = 128  # make this bigger
'''



x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')


def recurrent_neural_network(x):
    layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])),
             'biases': tf.Variable(tf.random_normal([n_classes]))}

    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, chunk_size])
    x = tf.split(0, n_chunks, x)

    lstm_cell = rnn_cell.BasicLSTMCell(rnn_size,state_is_tuple=True)
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    output = tf.matmul(outputs[-1], layer['weights']) + layer['biases']

    return output


def train_neural_network(x):
    global current_batch, up_to_subfile

    # Calculating mfccs vectors!!!
    audio_to_mfccs()

    prediction = recurrent_neural_network(x)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction, y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)     # default learning rate of 0.001

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # tf.initialize_all_variables()

        for epoch in range(hm_epochs):
            epoch_loss = 0

            for _ in range(int((4 * num_of_audio_in_language) / batch_size)):   # int(mnist.train.num_examples / batch_size)
                print('Ok\tepoch:', epoch, 'iter:', _)
                # epoch_x is of passed as shape: (1 X coeff X time). 1 is actually the batch_size.
                epoch_x, epoch_y = next_batch()    # mnist.train.next_batch(batch_size)
                # print('epoch_x:', epoch_x, '\n')

                # Case: passed the length of data set.
                if epoch_x is None and epoch_y is None:
                    break

                try:
                    epoch_x = epoch_x.reshape(batch_size, n_chunks, chunk_size)
                except:
                    print('reshape problem\tepoch:', epoch, 'iter:', _)
                    continue

                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c

            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))



        # Error said that it needs to be of size (3,5) and I was feeding it (?,3,5)
        # Trying to make the test file the right shape..
        '''
        test_audio = 'b0351.wav'
        test_mfcc = getMfccs(test_audio)
        test_input = []
        for i in range(NUM_OF_COEFF, chunk_size):
            # test_input =
        print('test_mfcc: ', test_mfcc)
        label = [1, 0, 0, 0]
        '''

        epoch_x, label = next_batch()


        try:
            print('Accuracy:',
              accuracy.eval({x: epoch_x.reshape((-1, n_chunks, chunk_size)), y: label}))  # accuracy.eval({x: mnist.test.images.reshape((-1, n_chunks, chunk_size)), y: mnist.test.labels}))
        except:
            print('Accuracy reshape problem.')

train_neural_network(x)