#copyright  spirit 
import re
import itertools
from collections import Counter
import numpy as np 
f=open('/home/wk15/20news/news-train.txt','r')
g=open('/home/wk15/20news/news-test.txt','r')
def clean_str(string):
    string=re.sub(r'[^A-Za-z0-9(),!?\'\`]',' ',string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", "  ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()
with f as file1:
    lines=f.readlines()
    train_labels=[]
    train_data=[]
    for raw in lines:
        strL,data=raw.split('\t')
        train_labels.append(strL)
        train_data.append(data)
with g as file2:
    lines=g.readlines()
    test_labels=[]
    test_data=[]
    for raw in lines:
        strL,data=raw.split('\t')
        test_data.append(data)
        test_labels.append(strL)
train=[clean_str(s).split(' ') for s in train_data]
test=[clean_str(s).split(' ') for s in test_data]
def word_padding(sentences,padding_word='#'):
    sentence_length=1000
    padded_sententces=[]
    for i in range(len(sentences)):
        sentence=sentences[i]
        num_padding=sentence_length-len(sentence)
        new_sententce=sentence+[padding_word]*num_padding
        if len(new_sententce)>1000:
            new_sententce=new_sententce[:1000]
        padded_sententces.append(new_sententce)
    return padded_sententces
train=word_padding(train)
test=word_padding(test)

train_target=[]
test_target=[]
for label in train_labels:
    if label=='0':
        train_target.append([1,0,0,0])
    elif label=='1':
        train_target.append([0,1,0,0])
    elif label=='2':
        train_target.append([0,0,1,0])
    else:
        train_target.append([0,0,0,1])
for label in test_labels:
    if label=='0':
        test_target.append([1,0,0,0])
    elif label=='1':
        test_target.append([0,1,0,0])
    elif label=='2':
        test_target.append([0,0,1,0])
    else:
        test_target.append([0,0,0,1])
vocab=[]
vocab.extend(train)
vocab.extend(test)
def build_data(sentence):
    word_counts=Counter(itertools.chain(*sentence))
    vocabulary_inv=[x[0] for x in word_counts.most_common()]
    vocabulary_inv=list(sorted(vocabulary_inv))
    vocabulary={x:i for i,x in enumerate(vocabulary_inv)}
    return [vocabulary,vocabulary_inv]
vocabulary,vocabulary_inv=build_data(vocab)

def build_input_data(sentences,labels,vocabulary):
    x=np.array([[vocabulary[word] for word in sentence]for sentence in sentences])
    y=np.array(labels)
    return [x,y]
x_train,y_train=build_input_data(train,train_target,vocabulary)
x_test,y_test=build_input_data(test,test_target,vocabulary)

import tensorflow as tf 
num_classes=4
x=tf.placeholder(tf.int32,[None,1000])
y=tf.placeholder(tf.float32,[None,num_classes])


dropout=tf.placeholder(tf.float32,name='dropout')

vocab_size=len(vocabulary)
embedding_size=64
sequence_length=1000
import numpy as np
import tensorflow as tf 
def add_layer(inputs,w,in_size,out_size,activation_function=None):
    Weights=tf.get_variable('w',shape=[128,4],initializer=tf.contrib.layers.xavier_initializer())
    biases=tf.Variable(tf.constant(0.1,shape=[4]))
    output=tf.nn.xw_plus_b(inputs,Weights,biases)
    return output
def add_conv_layer(inputs,filter_size,embedding_size,activation_function=None):
    Weights=tf.Variable(tf.truncated_normal([filter_size,embedding_size,1,64]))

    biases= tf.Variable(tf.constant(0.1, shape=[128]), name="b")
    
    Wx_plus_b=tf.nn.conv2d(inputs,Weights,strides=[1,1,1,1],padding='VALID',name='conv')
    
    out=tf.nn.relu(tf.nn.bias_add(Wx_plus_b,biases),name='relu')
    
    pooling=tf.nn.max_pool(out,
    ksize=[1,sequence_length-3+1,1,1],
    strides=[1,1,1,1],
    padding='VALID',
    name='pooling'
        )
    h_flat=tf.reshape(pooling,[-1,128])
    h_dropout=tf.nn.dropout(h_flat,dropout)
    return h_dropout,Weights

def add_embedding_layer(inputs,in_size,out_size,activation_function=None):
    Weights=tf.Variable(tf.random_uniform([in_size,out_size],-1.0,1.0))
    embedded_chars=tf.nn.embedding_lookup(Weights,inputs)
    embedded_chars_expanded=tf.expand_dims(embedded_chars,-1)
    return embedded_chars_expanded


embedded=add_embedding_layer(x,len(vocabulary),128)
l1,w=add_conv_layer(embedded,3,128,activation_function=tf.nn.relu)
output=add_layer(l1,w,128,4,activation_function=tf.nn.relu)
#output=add_layer(prediction,32,4,activation_function=tf.nn.relu)
pred=tf.argmax(output,1)








loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(output,y))

correct_predictions=tf.equal(pred,tf.argmax(y,1))

accuracy=tf.reduce_mean(tf.cast(correct_predictions,'float'))


optimizer=tf.train.AdamOptimizer(1e-3)
train_op=optimizer.compute_gradients(loss)

train_step=optimizer.apply_gradients(train_op)

with tf.Session() as sess:
    init=tf.initialize_all_variables()
    sess.run(init)
    for i in range(1000):
        shuffle_indices = np.random.permutation(np.arange(x_train.shape[0]))
        x_train=x_train[shuffle_indices]
        y_train=y_train[shuffle_indices]
        sess.run(train_step,feed_dict={x:x_train[:50],y:y_train[:50],dropout:0.5})
        print 'Iterim:',i,'  accuracy:',sess.run(accuracy,feed_dict={x:x_train[:50],y:y_train[:50],dropout:1.0}),'  loss:',sess.run(loss,feed_dict={x:x_train[:50],y:y_train[:50],dropout:1.0})
        if i%20==0:
            shuffle_indices = np.random.permutation(np.arange(x_test.shape[0]))
            x_test=x_test[shuffle_indices]
            y_test=y_test[shuffle_indices]
            print sess.run(accuracy,feed_dict={x:x_test[:50],y:y_test[:50],dropout:1.0})

