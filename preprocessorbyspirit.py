#copyright  spirit 
import re
import itertools
from collections import Counter
import numpy as np 
f=open('news-train.txt','r')
g=open('news-test.txt','r')
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
