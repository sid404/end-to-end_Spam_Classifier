# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 16:58:45 2021

@author: SID
"""
from flask import Flask,request,render_template
import pickle
import numpy as np
import nltk
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

filename='spam-classifier.pkl'
classifier=pickle.load(open(filename,'rb'))
cv=pickle.load(open('spam_classifier-vec.pkl','rb'))

app=Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data = [message]
        vect=cv.transform(data).toarray()
        my_prediction = classifier.predict(vect)
    return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True,use_reloader=False)

