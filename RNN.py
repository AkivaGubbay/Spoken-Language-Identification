# Imports:
from __future__ import print_function
import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from PrePorcessing import *
from time import time
# ========================================================== keep prob==========================

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