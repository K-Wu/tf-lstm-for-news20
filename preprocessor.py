
import numpy as np
import re
#import tensorflow as tf
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
        x_in_words.append(x_in_words[i][maximum_words_in_sentences-2:])
        labels_in_digits.append(labels_in_digits[i])
        x_in_words[i]=x_in_words[i][0:maximum_words_in_sentences]
    i=i+1
i=0
while i < len(x_test_in_words):
    if(len(x_test_in_words[i])>maximum_words_in_sentences):
        x_test_in_words.append(x_in_words[i][maximum_words_in_sentences-2:])
        labels_test_in_digits.append(labels_test_in_digits[i])
        x_test_in_words[i]=x_test_in_words[i][0:maximum_words_in_sentences]
    i=i+1
if len(x_in_words)!=len(labels_in_digits):
    print "not equal3"
#print x_in_words[0][0:100]
#print x_in_words[1][0:100]
#print x_in_words[2][0:100]
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
        labels_test_in_digits.append([0,0,0,1])
    elif labels_in_digits[i]=='1':
        labels_test_in_digits.append([0,0,1,0])
    elif labels_in_digits[i]=='2':
        labels_test_in_digits.append([0,1,0,0])
    elif labels_in_digits[i]=='3':
        labels_test_in_digits.append([1,0,0,0])
#randomly creating one-hot word representation in vec TODO:introduce word2vec model
vocabulary=dict()
for sentence in x_in_words:
    for word in sentence:
        if(vocabulary.has_key(word) is False):
            vocabulary[word]=len(vocabulary)+1
for sentence in x_test_in_words:
    for word in sentence:
        if(vocabulary.has_key(word) is False):
            vocabulary[word]=len(vocabulary)+1
#vectorize sentences and feed them into numpy array
x=[]
x_test=[]
#vocab_processor = learn.preprocessing.VocabularyProcessor(300)
#x = np.array(list(vocab_processor.fit_transform(x_in_words)))
#x_test = np.array(list(vocab_processor.fit_transform(x_test_in_words)))
for sentence in x_in_words:
    temp=[vocabulary[word] for word in sentence]
    for i in range(maximum_words_in_sentences-len(temp)):
        temp.append(0)
    x.append(temp)
for sentence in x_test_in_words:
    temp=[vocabulary[word] for word in sentence]
    for i in range(maximum_words_in_sentences-len(temp)):
        temp.append(0)
    x_test.append(temp)
for i,abc in enumerate(x):
    if len(abc)!=maximum_words_in_sentences:
	print i
x=np.array(x)
#print x.shape
#x_test=np.array(x_test)
labels_in_vec=np.array(labels_in_vec)
labels_test_in_vec=np.array(labels_test_in_vec)
