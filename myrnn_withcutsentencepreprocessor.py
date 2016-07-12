import numpy as np
import re
import tensorflow as tf


# from tensorflow
# from tensorflow.contrib import learn
def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"@", " ", string)
    return string.strip().lower()


# loading
examples = list(open("news-train.txt").readlines())
tests = list(open("news-test.txt").readlines())
labels_in_digits = [examples[i][0] for i in range(len(examples))]
labels_test_in_digits = [tests[i][0] for i in range(len(tests))]
x_in_string = [examples[i][1:] for i in range(len(examples))]
x_test_in_string = [tests[i][1:] for i in range(len(tests))]

# tokenize and cut sentences
x_in_words = []
x_test_in_words = []
for i in range(len(labels_in_digits)):
    x_in_words.append(clean_str(x_in_string[i]).split(" "))
for i in range(len(labels_test_in_digits)):
    x_test_in_words.append(clean_str(x_test_in_string[i]).split(" "))
i = 0
if len(x_in_words) != len(labels_in_digits):
    print "not equal2"
maximum_words_in_sentences = 350
i = 0
while i < len(x_in_words):
    if (len(x_in_words[i]) > maximum_words_in_sentences):
        x_in_words.append(x_in_words[i][maximum_words_in_sentences - 2:])
        labels_in_digits.append(labels_in_digits[i])
        x_in_words[i] = x_in_words[i][0:maximum_words_in_sentences]
    i = i + 1
i = 0
while i < len(x_test_in_words):
    if (len(x_test_in_words[i]) > maximum_words_in_sentences):
        x_test_in_words.append(x_test_in_words[i][maximum_words_in_sentences - 2:])
        labels_test_in_digits.append(labels_test_in_digits[i])
        x_test_in_words[i] = x_test_in_words[i][0:maximum_words_in_sentences]
    i = i + 1

# vectorize labels
labels_in_vec = []
labels_test_in_vec = []
for i in range(len(labels_in_digits)):
    if labels_in_digits[i] == '0':
        labels_in_vec.append([0, 0, 0, 1])
    elif labels_in_digits[i] == '1':
        labels_in_vec.append([0, 0, 1, 0])
    elif labels_in_digits[i] == '2':
        labels_in_vec.append([0, 1, 0, 0])
    elif labels_in_digits[i] == '3':
        labels_in_vec.append([1, 0, 0, 0])
for i in range(len(labels_test_in_digits)):
    if labels_test_in_digits[i] == '0':
        labels_test_in_vec.append([0, 0, 0, 1])
    elif labels_test_in_digits[i] == '1':
        labels_test_in_vec.append([0, 0, 1, 0])
    elif labels_test_in_digits[i] == '2':
        labels_test_in_vec.append([0, 1, 0, 0])
    elif labels_test_in_digits[i] == '3':
        labels_test_in_vec.append([1, 0, 0, 0])

# randomly creating one-hot word representation in vec TODO:introduce word2vec model
vocabulary = dict()
vocabulary[''] = 0
for sentence in x_in_words:
    for word in sentence:
        if (vocabulary.has_key(word) is False):
            vocabulary[word] = len(vocabulary)
for sentence in x_test_in_words:
    for word in sentence:
        if (vocabulary.has_key(word) is False):
            vocabulary[word] = len(vocabulary)

# vectorize sentences and feed them into numpy array
x = []
seqlen_x = []
x_test = []
seqlen_x_test = []

for sentence in x_in_words:
    temp = [vocabulary[word] for word in sentence]
    seqlen_x.append(len(temp))
    for i in range(maximum_words_in_sentences - len(temp)):
        temp.append(0)
    x.append(temp)
for sentence in x_test_in_words:
    temp = [vocabulary[word] for word in sentence]
    seqlen_x_test.append(len(temp))
    for i in range(maximum_words_in_sentences - len(temp)):
        temp.append(0)
    x_test.append(temp)
x = np.array(x)
x_test = np.array(x_test)
seqlen_x = np.array(seqlen_x)
seqlen_x_test = np.array(seqlen_x_test)
labels_in_vec = np.array(labels_in_vec)
labels_test_in_vec = np.array(labels_test_in_vec)

# constructing the computing graph
num_classes = 4
num_filters = 32
vocab_size = len(vocabulary)
embedding_size = 300
input_x = tf.placeholder(tf.int32, [None, maximum_words_in_sentences], name="input_x")
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
# input_x_expanded = tf.expand_dims(input_x, -1)
# Embedding layer
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=False,
        name="W")
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)


def cnn():
    filter_sizes = [3, 5, 5, 7]
    pooled_outputs = []
    for i, filter_size in enumerate(filter_sizes):
        with tf.name_scope("conv-maxpoor-%s" % filter_size):
            # conv layer
            filter_shape = [filter_size, 100, 1, num_filters]
            W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
            b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
            conv = tf.nn.conv2d(
                embedded_chars_expanded,
                W,
                strides=[1, 1, 1, 1],
                padding="VALID",
                name="conv")
            # Apply nonlinearity
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # TODO:Check different activation methods
            # Maxpooling
            pooled = tf.nn.max_pool(
                h,
                ksize=[1, maximum_words_in_sentences - filter_size + 1, 1, 1],
                strides=[1, 1, 1, 1],
                padding='VALID',
                name="pool"
            )
            pooled_outputs.append(pooled)
    # Combine all the pooled features
    num_filters_total = num_filters * len(filter_sizes)
    h_pool = tf.concat(3, pooled_outputs)
    h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
    # dropout
    with tf.name_scope("dropout"):
        h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)


# Parameters
learning_rate = 0.01
training_iters = 20000
batch_size = 56
display_step = 10

# Network Parameters

n_hidden = 64  # hidden layer num of features
# tf Graph input

# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def dynamicRNN(x, seqlen):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    #x = tf.transpose(x, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    #x = tf.reshape(x, [-1, maximum_words_in_sentences])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(1, maximum_words_in_sentences, x)
    x = [tf.squeeze(x_, [1]) for x_ in x]
    # Define a lstm cell with tensorflow
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=0.3)

    tm_cell = tf.nn.rnn_cell.DropoutWrapper(
               lstm_cell, output_keep_prob= 0.35)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    # calculation.
    #print tf.shape(embedded_chars)
    outputs, states = tf.nn.rnn(cell, x, dtype=tf.float32,
                                sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e, if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    #outputs = tf.pack(outputs)
    #outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    #batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    #index = tf.range(0, batch_size) * maximum_words_in_sentences + (seqlen - 1)
    # Indexing
    #outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
    #output = tf.reshape(tf.concat(1, outputs), [-1, embedding_size])
    #outputs of tf.nn.rnn  (maximum_words,?,n_hidden)
    #outputs=tf.split(1,tf.shape(outputs)[1],outputs)
    #outputs=[tf.squeeze(output,[1]) for output in outputs]
    # Linear activation, using outputs computed above
    #print tf.shape(states)
    #outputs=tf.reduce_mean(outputs,0)
    outputs=tf.pack(outputs)
    outputs= tf.transpose(outputs, [1,0,2])
    return outputs

input_for_cnn = dynamicRNN(embedded_chars, seqlen)
input_for_cnn = tf.expand_dims(input_for_cnn, -1)
filter_sizes = [3, 5, 5, 7]
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpoor-%s" % filter_size):
        # conv layer
        filter_shape = [filter_size, n_hidden, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            input_for_cnn,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        # Apply nonlinearity
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")  # TODO:Check different activation methods
        # Maxpooling
        pooled = tf.nn.max_pool(
            h,
            ksize=[maximum_words_in_sentences- filter_size + 1, n_hidden, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool"
        )
        pooled_outputs.append(pooled)
# Combine all the pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(3, pooled_outputs)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
# dropout
with tf.name_scope("dropout"):
    h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)


# Final scores and predictions
with tf.name_scope("output"):
    W=tf.get_variable(
       "W",        shape=[num_filters_total,num_classes],
       initializer=tf.contrib.layers.xavier_initializer())
    b=tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b")
    scores=tf.nn.xw_plus_b(h_drop,W,b,name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")
# Calculate Mean cross-entropy loss
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(scores, input_y)
    loss = tf.reduce_mean(losses)
# Accuracy
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

optimizer = tf.train.AdamOptimizer(1e-3)
train_op = optimizer.compute_gradients(loss)
train_step = optimizer.apply_gradients(train_op)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(50000):
        shuffle_indices = np.random.permutation(np.arange(x.shape[0]))
        x = x[shuffle_indices]
        labels_in_vec = labels_in_vec[shuffle_indices]
        seqlen_x = seqlen_x[shuffle_indices]
        sess.run(train_step, feed_dict={input_x: x[:batch_size], input_y: labels_in_vec[:batch_size], seqlen: seqlen_x[:batch_size], dropout_keep_prob: 0.5})
        print 'Iterim:', i, '  accuracy:', sess.run(accuracy, feed_dict={input_x: x[:batch_size],input_y: labels_in_vec[:batch_size],seqlen: seqlen_x[:batch_size], dropout_keep_prob:1.0}), \
            '  loss:', sess.run(
            loss, feed_dict={input_x: x[:50], input_y: labels_in_vec[:50], seqlen: seqlen_x[:50], dropout_keep_prob: 1.0})
        if i % 100 == 0:
            shuffle_test_indices = np.random.permutation(np.arange(x_test.shape[0]))
            x_test = x_test[shuffle_test_indices]
            labels_test_in_vec = labels_test_in_vec[shuffle_test_indices]
            print sess.run(accuracy,
                           feed_dict={input_x: x_test[:], input_y: labels_test_in_vec[:], dropout_keep_prob: 1.0, seqlen: seqlen_x_test[:]})
