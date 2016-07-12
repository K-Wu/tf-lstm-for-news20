import numpy as np
import re
import tensorflow as tf
#from tensorflow.contrib import learn
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
#loading
examples=list(open("news-train.txt").readlines())
tests=list(open("news-test.txt").readlines())
labels_in_digits=[examples[i][0] for i in range(len(examples))]
labels_test_in_digits=[tests[i][0] for i in range(len(tests))]
x_in_string=[examples[i][1:] for i in range(len(examples))]
x_test_in_string=[tests[i][1:] for i in range(len(tests))]
if len(labels_in_digits)!=len(x_in_string):
    print "not equal 1"

maximum_words_in_sentences=300
#tokenize and cut sentences
x_in_words=[]
x_test_in_words=[]
for i in range(len(labels_in_digits)):
    x_in_words.append(clean_str(x_in_string[i]).split(" "))
for i in range(len(labels_test_in_digits)):
    x_test_in_words.append(clean_str(x_test_in_string[i]).split(" "))
i=0
if len(x_in_words)!=len(labels_in_digits):
    print "not equal2"
while i < len(x_in_words):
    if(len(x_in_words[i])>maximum_words_in_sentences):
        x_in_words.append(x_in_words[i][298:])
        labels_in_digits.append(labels_in_digits[i])
        x_in_words[i]=x_in_words[i][0:300]
    i=i+1
i=0
while i < len(x_test_in_words):
    if(len(x_test_in_words[i])>maximum_words_in_sentences):
        x_test_in_words.append(x_test_in_words[i][298:])
        labels_test_in_digits.append(labels_test_in_digits[i])
        x_test_in_words[i]=x_test_in_words[i][0:300]
    i=i+1
if len(x_in_words)!=len(labels_in_digits):
    print "not equal3"
if len(x_test_in_words)!=len(labels_test_in_digits):
    print "not equal3 test"
#vectorize labels
labels_in_vec=[]
labels_test_in_vec=[]
for i in range(len(labels_in_digits)):
    if labels_in_digits[i]=='0':
        labels_in_vec.append([0,0,0,1])
    elif labels_in_digits[i]=='1':
        labels_in_vec.append([0,0,1,0])
    elif labels_in_digits[i]=='2':
        labels_in_vec.append([0,1,0,0])
    elif labels_in_digits[i]=='3':
        labels_in_vec.append([1,0,0,0])
for i in range(len(labels_test_in_digits)):
    if labels_test_in_digits[i]=='0':
        labels_test_in_vec.append([0,0,0,1])
    elif labels_test_in_digits[i]=='1':
        labels_test_in_vec.append([0,0,1,0])
    elif labels_test_in_digits[i]=='2':
        labels_test_in_vec.append([0,1,0,0])
    elif labels_test_in_digits[i]=='3':
        labels_test_in_vec.append([1,0,0,0])
#randomly creating one-hot word representation in vec TODO:introduce word2vec model
vocabulary=dict()
vocabulary['']=0
for sentence in x_in_words:
    for word in sentence:
        if(vocabulary.has_key(word) is False):
            vocabulary[word]=len(vocabulary)
for sentence in x_test_in_words:
    for word in sentence:
        if(vocabulary.has_key(word) is False):
            vocabulary[word]=len(vocabulary)
#vectorize sentences and feed them into numpy array
x=[]
x_test=[]
#vocab_processor = learn.preprocessing.VocabularyProcessor(300)
#x = np.array(list(vocab_processor.fit_transform(x_in_words)))
#x_test = np.array(list(vocab_processor.fit_transform(x_test_in_words)))
for sentence in x_in_words:
    temp=[vocabulary[word] for word in sentence]
    for i in range(300-len(temp)):
        temp.append(0)
    x.append(temp)
for sentence in x_test_in_words:
    temp=[vocabulary[word] for word in sentence]
    for i in range(300-len(temp)):
        temp.append(0)
    x_test.append(temp)
x=np.array(x)
#print x.shape
x_test=np.array(x_test)
labels_in_vec=np.array(labels_in_vec)
labels_test_in_vec=np.array(labels_test_in_vec)
if labels_in_vec.shape[0]!=x.shape[0]:
    print "not equal 4 "
if labels_test_in_vec.shape[0] !=x_test.shape[0]:
    print "not equal 4 test"
    print "labels_test_in_vec"
    print labels_test_in_vec.shape[0]
    print "labels_test_in_digts"
    print len(labels_test_in_digits)
    print "x_test"
    print x_test.shape[0]
    print "x_test_in_words"
    print len(x_test_in_words)
#constructing the computing graph
sequence_length=maximum_words_in_sentences
num_classes=4
num_filters=32
vocab_size=len(vocabulary)
embedding_size=100
input_x=tf.placeholder(tf.int32,[None,sequence_length],name="input_x")
input_y=tf.placeholder(tf.float32,[None,num_classes],name="input_y")
dropout_keep_prob=tf.placeholder(tf.float32,name="dropout_keep_prob")
filter_sizes=[5,5,5,7]
#input_x_expanded = tf.expand_dims(input_x, -1)
# Embedding layer
with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(
        tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),trainable=False,
        name="W")
    embedded_chars = tf.nn.embedding_lookup(W, input_x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)
pooled_outputs=[]
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpoor-%s" % filter_size):
        #conv layer
        filter_shape=[filter_size,100,1,num_filters]
        W=tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1),name="W")
        b=tf.Variable(tf.constant(0.1,shape=[num_filters]),name="b")
        conv=tf.nn.conv2d(
            embedded_chars_expanded,
                    W,
            strides=[1,1,1,1],
            padding="VALID",
            name="conv")
        #Apply nonlinearity
        h=tf.nn.relu(tf.nn.bias_add(conv,b),name="relu")#TODO:Check different activation methods
        #Maxpooling
        pooled=tf.nn.max_pool(
            h,
            ksize=[1,sequence_length-filter_size+1,1,1],
            strides=[1,1,1,1],
            padding='VALID',
            name="pool"
        )
        pooled_outputs.append(pooled)
#Combine all the pooled features
num_filters_total=num_filters*len(filter_sizes)
h_pool=tf.concat(3,pooled_outputs)
h_pool_flat=tf.reshape(h_pool,[-1,num_filters_total])
#dropout
with tf.name_scope("dropout"):
    h_drop=tf.nn.dropout(h_pool_flat,dropout_keep_prob)
#Final scores and predictions
with tf.name_scope("output"):
    W=tf.get_variable(
        "W",
        shape=[num_filters_total,num_classes],
        initializer=tf.contrib.layers.xavier_initializer())
    b=tf.Variable(tf.constant(0.1,shape=[num_classes]),name="b")
    scores=tf.nn.xw_plus_b(h_drop,W,b,name="scores")
    predictions=tf.argmax(scores,1,name="predictions")
#Calculate Mean cross-entropy loss
with tf.name_scope("loss"):
    losses=tf.nn.softmax_cross_entropy_with_logits(scores,input_y)
    loss=tf.reduce_mean(losses)
#Accuracy
with tf.name_scope("accuracy"):
    correct_predictions=tf.equal(predictions,tf.argmax(input_y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")


optimizer=tf.train.AdamOptimizer(1e-3)
train_op=optimizer.compute_gradients(loss)
train_step=optimizer.apply_gradients(train_op)
print x.shape
with tf.Session() as sess:
    init=tf.initialize_all_variables()
    sess.run(init)
    for i in range(50000):
        shuffle_indices = np.random.permutation(np.arange(x.shape[0]))
        x=x[shuffle_indices]
        labels_in_vec=labels_in_vec[shuffle_indices]
        sess.run(train_step,feed_dict={input_x:x[:50],input_y:labels_in_vec[:50],dropout_keep_prob:0.3})
        print 'Iterim:',i,'  accuracy:',sess.run(accuracy,feed_dict={input_x:x[:50],input_y:labels_in_vec[:50],dropout_keep_prob:1.0}),'  loss:',sess.run(loss,feed_dict={input_x:x[:50],input_y:labels_in_vec[:50],dropout_keep_prob:1.0})
        if i%100==0:
            shuffle_test_indices = np.random.permutation(np.arange(x_test.shape[0]))
            x_test=x_test[shuffle_test_indices]
            labels_test_in_vec=labels_test_in_vec[shuffle_test_indices]
            print sess.run(accuracy,feed_dict={input_x:x_test[:],input_y:labels_test_in_vec[:],dropout_keep_prob:1.0})