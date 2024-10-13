import os
import pickle
import jieba
import numpy as np
from gensim.models import Word2Vec
from tqdm import tqdm

email_list=input("请输入邮箱内容：")
words= jieba.cut(email_list)

with open('stopwords.txt', 'r', encoding='UTF-8') as f:
    stopwords = f.read().strip().split('\n')
X=[]

for w in words:
        if w not in stopwords:
            X.append(w)
if os.path.exists('./word2vec.model'):
    w2c = Word2Vec.load('word2vec.model')

with tqdm(total=len(X),desc='word2vec融合中:') as pbar:
    vec = np.zeros((1,100))
    for word in X:
        try:
            vec = vec +w2c.wv.get_vector(word).reshape((1,100))
        except KeyError:
            pbar.update(1)
            continue
    pbar.update(1)
    X = vec

with open('BernoulliNB.pickle', 'rb') as f:
    nb = pickle.load(f)
a=nb.predict(X)
print(a)