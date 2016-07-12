import numpy as np
import time
import cPickle as pickle
#This file is for loading GloVe pretrained .txt data
def load_word2vec(filename):
    #store data in two variables, one is word:rowid, the second is a two dimensional numpy array which stores word vector in the corresponding row
    word_rowid_dictionary = {} #skip information on first line
    vector_lookuptable_ndarray=[]
    fin= open(filename)
    word_rowid_dictionary['']=0
    vector_lookuptable_ndarray.append([0 for i in range(300)])
    for line in fin:
        items = line.replace('\r','').replace('\n','').split(' ')
        if len(items) < 10: continue
        word = items[0]
        word_rowid_dictionary[word] = len(vector_lookuptable_ndarray)
        vector_lookuptable_ndarray.append([float(i) for i in items[1:] if len(i) > 1])
    vector_lookuptable_ndarray=np.array(vector_lookuptable_ndarray)
    f1=file('word_rowid_dict.pkl','wb')
    pickle.dump(word_rowid_dictionary,f1,True)
    np.save("glove.840B.300d.vec.npy",vector_lookuptable_ndarray)
    f1.close()
   #return word_rowid_dictionary, vector_lookuptable_ndarray
a=time.time()
load_word2vec("glove.840B.300d/glove.840B.300d.txt")
b=time.time()
f2=file('word_rowid_dict.pkl','rb')
word_rowid_dictionray=pickle.load(f2)
vector_lookuptable_ndarray=np.load("glove.840B.300d.vec.npy")
c=time.time()
f2.close()
print "load txt to np.array and dict and store", (b-a)
print "load dict and np.array from file", (c-b)
print vector_lookuptable_ndarray.shape
