# -*- coding: utf-8 -*-
"""
@author: FARAZ AHMAD
"""
#Use data for testing
import pandas as pd
import numpy as np
import itertools
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('fake_or_real_news.csv')
y = df.label
df = df.drop('label', axis=1)
X_train, X_test, y_train, y_test = train_test_split(df['text'], y, test_size=0.33, random_state=53)
count_vectorizer = CountVectorizer(stop_words='english')
count_train = count_vectorizer.fit_transform(X_train)
count_test = count_vectorizer.transform(X_test)
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
tfidf_train = tfidf_vectorizer.fit_transform(X_train)
tfidf_test = tfidf_vectorizer.transform(X_test)
#mn_count_clf = MultinomialNB(alpha=0.1)
mn_count_clf = SGDClassifier()
mn_count_clf.fit(count_train, y_train)
pred = mn_count_clf.predict(count_test)

news=input("Enter a piece of news: ")
from io import StringIO
TESTDATA = StringIO(news)
d=pd.read_csv(TESTDATA,header=None)
sentence=d[0]
ct= count_vectorizer.transform(sentence)
newsprep= mn_count_clf.predict(ct)
print(newsprep)
