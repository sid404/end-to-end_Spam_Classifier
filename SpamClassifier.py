# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:09:53 2021

@author: SID
"""
import pandas as pd
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pickle

df=pd.read_csv('SMSSpamCollection.csv',delimiter='\t')
df.columns=['Label','Message']


X=df['Message']
Y=df['Label']

corpus=[]
lem=WordNetLemmatizer()
for i in range(len(X)):
    cln=re.sub('[^a-zA-Z]',' ',X[i])
    cln=cln.lower()
    cln=cln.split(' ')
    cln=[lem.lemmatize(word) for word in cln if word not in stopwords.words('english')]
    cln=' '.join(cln)
    corpus.append(cln)
    
from sklearn.feature_extraction.text import TfidfVectorizer
vec=TfidfVectorizer()
X=vec.fit_transform(corpus).toarray()

Y=pd.get_dummies(Y,drop_first=True)

from imblearn.over_sampling import RandomOverSampler
os=RandomOverSampler()
X,Y=os.fit_resample(X,Y)

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)

from sklearn.naive_bayes import MultinomialNB
mnb=MultinomialNB()
mnb.fit(X_train,Y_train)
pred=mnb.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(Y_test,pred))
print("\n")
print(classification_report(Y_test,pred))

filename='spam-classifier.pkl'
pickle.dump(mnb,open(filename,'wb'))

pickle.dump(vec,open('spam_classifier-vec.pkl','wb'))