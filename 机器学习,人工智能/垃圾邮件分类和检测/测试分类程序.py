import os
import jieba
import numpy as np
import pickle
import re
from gensim.models import Word2Vec

def preprocess_email(email_content, stopwords):
    email_content = re.sub(r"[^\u4e00-\u9fff]", "", email_content)
    split_words = jieba.cut(email_content)
    split_words_list = [w for w in split_words if w not in stopwords]
    return split_words_list

def load_models():
    if os.path.exists('/word2vec.model'):
        model = Word2Vec.load('word2vec.model')
    else:
        raise FileNotFoundError("Word2Vec模型文件不存在")

    with open("BernoulliNB.pickle", "rb") as f:
        nb = pickle.load(f)
    return model, nb

def vectorize_text(text, model):
    vec = np.zeros((1, 100))
    for word in text:
        try:
            vec += model.wv[word].reshape((1, 100))
        except KeyError:
            continue
    return np.squeeze(vec)

def classify_email(text, model, nb, stopwords):
    split_words_list = preprocess_email(text, stopwords)
    vec = vectorize_text(split_words_list, model)
    prediction = nb.predict([vec])
    return prediction[0]

def main():
    # 直接获取用户输入的邮件内容
    email_content = input("请输入邮件内容：")

    model, nb = load_models()

    with open("stopwords.txt", "r", encoding='utf-8') as f:
        stopwords = f.read().strip().split('\n')

    result = classify_email(email_content, model, nb, stopwords)

    if result == 0:
        print("邮件分类结果：垃圾邮件")
    else:
        print("邮件分类结果：非垃圾邮件")

if __name__ == "__main__":
    main()
