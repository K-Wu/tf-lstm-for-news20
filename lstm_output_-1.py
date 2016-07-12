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
num_epoch=50
num_batches=1000
dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
# input_x_expanded = tf.expand_dims(input_x, -1)
# Embedding layer
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), trainable=False,
        name="W")
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

# Parameters
initial_learning_rate = 0.001
decay_rate=0.98
training_iters = 20000
batch_size = 56
display_step = 10

# Network Parameters

n_hidden = 96  # hidden layer num of features
# tf Graph input

# A placeholder for indicating each sequence length
seqlen = tf.placeholder(tf.int32, [None])
learning_rate=tf.placeholder(tf.float32, [])
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_hidden]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_hidden]))
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
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1)

    tm_cell = tf.nn.rnn_cell.DropoutWrapper(
               lstm_cell, output_keep_prob= 0.35)
    # cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * 2)
    # calculation. TODO:implement 2-layer
    outputs, states = tf.nn.rnn(tm_cell, x, dtype=tf.float32,
                                sequence_length=seqlen)
    #batch_size_=outputs[0].get_shape()[0]
    #batch_size_=tf.cast(batch_size_, int)
    #outputs=tf.reduce_mean(outputs,0)
    outputs=tf.split(1, batch_size, outputs)
    outputs=[tf.reshape(output,[-1,n_hidden]) for output in outputs]
    for i in range(batch_size):
        outputs[i]=tf.nn.xw_plus_b(outputs[i],weights['out'],biases['out'],name="linear")
        outputs[i]=tf.tanh(outputs[i])
    outputs=tf.pack(outputs)#the [1] with length of batch_size becomes [0] now
    outputs=tf.reduce_max(outputs,1)#change to 1 accordingly
    return outputs

lstm_output = dynamicRNN(embedded_chars, seqlen)


# dropout
#with tf.name_scope("dropout"):
#    h_drop = tf.nn.dropout(lstm_output, dropout_keep_prob)


# Final scores and predictions
with tf.name_scope("output"):
    W=tf.get_variable(
       "W",        shape=[n_hidden,num_classes])
    b=tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b")
    scores=tf.nn.xw_plus_b(lstm_output,W,b,name="scores")
    predictions = tf.argmax(scores, 1, name="predictions")
# Calculate Mean cross-entropy loss
with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(scores, input_y)
    loss = tf.reduce_mean(losses)
# Accuracy
with tf.name_scope("accuracy"):
    correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = optimizer.compute_gradients(loss)
train_step = optimizer.apply_gradients(train_op)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for e in range(num_epoch):
        for i in range(num_batches):
            shuffle_indices = np.random.permutation(np.arange(x.shape[0]))
            x = x[shuffle_indices]
            labels_in_vec = labels_in_vec[shuffle_indices]
            seqlen_x = seqlen_x[shuffle_indices]
            sess.run(train_step, feed_dict={input_x: x[:batch_size], input_y: labels_in_vec[:batch_size], seqlen: seqlen_x[:batch_size], dropout_keep_prob: 0.5,learning_rate:initial_learning_rate * (decay_rate ** e)})
            print 'Epoch', e, 'Iterim:', i, '  accuracy:', sess.run(accuracy, feed_dict={input_x: x[:batch_size],input_y: labels_in_vec[:batch_size],seqlen: seqlen_x[:batch_size], dropout_keep_prob:1.0}), \
                '  loss:', sess.run(
                loss, feed_dict={input_x: x[:batch_size], input_y: labels_in_vec[:batch_size], seqlen: seqlen_x[:batch_size], dropout_keep_prob: 1.0})
            if i % 100 == 0:
                shuffle_test_indices = np.random.permutation(np.arange(x_test.shape[0]))
                x_test = x_test[shuffle_test_indices]
                labels_test_in_vec = labels_test_in_vec[shuffle_test_indices]
                print sess.run(accuracy,
                               feed_dict={input_x: x_test[:batch_size], input_y: labels_test_in_vec[:batch_size], dropout_keep_prob: 1.0, seqlen: seqlen_x_test[:batch_size]})
