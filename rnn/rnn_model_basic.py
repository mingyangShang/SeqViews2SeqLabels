import tensorflow as tf
from tensorflow.contrib import rnn
from model_data import read_data
import numpy as np
import csv

# from tensorflow.examples.tutorials.mnist import input_data

tf.flags.DEFINE_string('model_path', '/home1/shangmingyang/data/3dmodel/model/model-vggfeature-50epoch/droupout-0.9-100epoch.ckpt', 'file path for saving model')
tf.flags.DEFINE_string('data_path', '/home3/lhl/tensorflow-vgg-master/feature', 'file dir for saving features and labels')

tf.flags.DEFINE_string('acc_result_file', '/home1/shangmingyang/data/3dmodel/acc_result_sigmoid.csv', 'result save file')

tf.flags.DEFINE_float('keep_prob', 1.0, 'keep probability for rnn cell')
tf.flags.DEFINE_integer("hidden_size", 128, "rnn cell hidden state")
tf.flags.DEFINE_float("forget_biases", 1.0, "lstm cell forget biases")

# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

FLAGS = tf.flags.FLAGS

#model_data = read_data(FLAGS.data_path)

learning_rate = 0.0001
training_iters = 3183 * 100
batch_size = 10
display_step = 100
save_step =  3183 * 1
need_save = True

# Network Parameters
n_steps = 12 # timesteps
n_input = 4096 # model_data data input (img shape: 28*28)
#n_hidden = 128 # hidden layer num of features
n_classes = 40 # model_data total classes (0-9 digits)

model_data = read_data(FLAGS.data_path)

x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])
p = tf.placeholder("float", shape=())

weights = {
    'out': tf.Variable(tf.random_normal([FLAGS.hidden_size, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    x_dropout = [tf.nn.dropout(xi, FLAGS.keep_prob) for xi in x]
    lstm_cell = rnn.BasicLSTMCell(FLAGS.hidden_size, forget_bias=FLAGS.forget_biases)
    #lstm_cell = rnn.GRUCell(FLAGS.hidden_size)
    outputs, states = rnn.static_rnn(lstm_cell, x_dropout, dtype=tf.float32)
    equal_output_hidden = tf.equal(outputs[-1], states)
    return tf.matmul(outputs[-1], weights['out']) + biases['out'], equal_output_hidden

def BiRNN(x, weights, biases):
    x = tf.unstack(x, n_steps, 1)
    #lstm_fw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_fw_cell = rnn.GRUCell(FLAGS.hidden_size)
    #lstm_bw_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.GRUCell(FLAGS.hidden_size)
    try:
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    except Exception:
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    return tf.nn.softmax(tf.matmul(outputs[-1], weights['out']) + biases['out'])

def main(unused_argv):
    x = tf.placeholder("float", [None, n_steps, n_input])
    y = tf.placeholder("float", [None, n_classes])
    p = tf.placeholder("float", shape=())

    pred, equal_output_hidden = RNN(x, weights, biases)
    #pred = BiRNN(x, weights, biases)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    output_label = tf.argmax(pred, 1)
    #output_label_prob = tf.reduce_max(pred, reduction_indices=1)
    #gb_label_prob = pred[np.arange(800), tf.argmax(y, 1)]

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    output_acc = tf.reduce_mean(tf.cast(equal_output_hidden, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    #saver = tf.train.Saver()

    acc_test_list = [FLAGS.keep_prob, FLAGS.hidden_size, FLAGS.forget_biases]

    # Launch the graph
    config = tf.ConfigProto()
    #config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    with tf.Session(config=config) as sess:
        sess.run(init)
        #saver.restore(sess, FLAGS.model_path)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size <= training_iters:
            batch_x, batch_y = model_data.train.next_batch(batch_size, as_sequence=False)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, p: FLAGS.keep_prob})
            if step % display_step == 0:
                # Calculate batch accuracy
                acc, equal = sess.run([accuracy, equal_output_hidden], feed_dict={x: batch_x, y: batch_y, p: 1.0})
                # Calculate batch loss
                loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                    "{:.6f}".format(loss) + ", Training Accuracy= " + \
                    "{:.5f}".format(acc) + ", outputEqualHidden=%f"%(equal))
            if step % save_step == 0 and need_save:
                #do test
                test_data = model_data.test.fcs.reshape((-1, n_steps, n_input))
                test_label = model_data.test.labels
                acc = sess.run(accuracy, feed_dict={x: test_data, y: test_label, p: 1.0})
                #saver.save(sess, FLAGS.model_path)
                print("acc:", acc)
                acc_test_list.append(acc)
            step += 1

        print("Optimization Finished!")

        # Calculate accuracy for 128 model_data test images
        # test_len = 256
        test_data = model_data.test.fcs.reshape((-1, n_steps, n_input))
        test_label = model_data.test.labels
        acc, label, pred = sess.run([accuracy, output_label, pred], feed_dict={x:test_data, y:test_label, p: 1.0})
        #output_label_prob = pred[np.arange(np.shape(test_label)[0]), label]
        #gb_label_prob = pred[np.arange(np.shape(test_label)[0]), np.argmax(test_label, 1)]
        print("Testing Accuracy:", acc)
        #print("labels:", label)
        #print("output label prob:", output_label_prob)
        #print("gb label prob:", gb_label_prob)
        print("keep_prob: %f " %(FLAGS.keep_prob))

        #label_list, output_prob_list, gb_prob_list = label.tolist(), output_label_prob.tolist(), gb_label_prob.tolist()
        #wrong_index_list, test_label_list = [], np.argmax(test_label, 1).tolist()
        #for i in xrange(len(label_list)):
            #if label_list[i] != test_label_list[i]:
                #wrong_index_list.append(i)

        #data = np.zeros(shape=[len(test_label_list), 4])
        #with open('prob.txt', 'w') as f:
            #for i in xrange(len(test_label_list)):
                #data[i] = np.array([label_list[i], test_label_list[i], output_prob_list[i], gb_prob_list[i]])

        #np.save('classify_result', data)
                #f.write(','.join([str(label_list[i]), str(output_prob_list[i]), str(gb_prob_list[i])]) + '\n')
        print('results list:', acc_test_list)
        with open(FLAGS.acc_result_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow(acc_test_list)
        #with open('result-1000epoch.txt', 'w') as f:
            #f.write('\n'.join([str(acc) for acc in acc_test_list]))


if __name__ == '__main__':
    tf.app.run()

