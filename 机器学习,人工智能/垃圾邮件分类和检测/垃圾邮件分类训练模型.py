import os
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from tqdm import tqdm
import re

with open('trec06c/full/index','r') as f:
    data = f.read().strip()

datas = data.split('\n')
y= []
X= []
for i in tqdm(range(len(datas)),desc='读取中'):
    d =datas[i]
    email_type,email_url = d.split(' ')
    y.append(email_type)
    email_url = email_url.replace('..','')
    with open('./trec06c{}'.format(email_url),'r',encoding='gb2312',errors='ignore') as f:
        text = f.read()
    text = re.sub(r"[^\u4e00-\u9fff]", "", text)
    text= re.sub(
        "[0-9a-zA-Z\-\s+\.\!\/_,$%^*\(\)\+(+\"\')]+|[+——！，。？、~@#￥%……&*（）<>\[\]:：★◆【】《》;；=?？]+", "", text)
    split_words = jieba.cut(text)
    split_words_list = []

    with open('stopwords.txt','r',encoding='UTF-8') as f:
        stopwords = f.read().strip().split('\n')
    for w in split_words:
        split_words_list.append(w)
    X.append(split_words_list)
if os.path.exists('./word2vec.model'):
    w2c = Word2Vec.load('word2vec.model')
else:
    w2c = Word2Vec(sentences=X, min_count=1, vector_size=100, workers=4)
    w2c.train(X, total_examples=w2c.corpus_count, epochs=w2c.epochs)
    w2c.save('word2vec.model')

with tqdm(total=len(X),desc='word2vec融合中:') as pbar:
    for i,x in enumerate(X):
        vec = np.zeros((1,100))
        for word in x:
            try:
                vec = vec +w2c.wv.get_vector(word).reshape((1,100))
            except KeyError:
                pbar.update(1)
                continue
        pbar.update(1)
        X[i] = vec

for i,e in enumerate(y):
    if e == 'spam':
        y[i] = 0
    else:
        y[i] = 1

X = np.squeeze(np.array(X))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=4)

import pickle

nb = GaussianNB()
nb.fit(X_train,y_train)
print(nb.score(X_test,y_test))
with open('GaussianNB.pickle','wb') as f:
    pickle.dump(nb,f)
# nb = MultinomialNB()
# nb.fit(X_train,y_train)
# print(nb.score(X_test,y_test))
# with open('MultinomialNB.pickle','wb') as f:
#     pickle.dump(nb,f)
nb = BernoulliNB()
nb.fit(X_train,y_train)
print(nb.score(X_test,y_test))
with open('BernoulliNB.pickle','wb') as f:
    pickle.dump(nb,f)
