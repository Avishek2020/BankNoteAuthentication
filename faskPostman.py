#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from flask import Flask, request
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

pickle_in = open('/home/avishek/Jupyter_Notebooks/BankNoteAuthentication/classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Hello world"



@app.route('/predict')
def predict_note_authentication_local():
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy  = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness, curtosis,entropy]])
    return "The predicted Value is "+ str(prediction)
    
@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The predicted Values for the csv is "+ str(list(prediction))


   
    
if __name__=='__main__':
    app.run('127.0.0.1')
    

#http://127.0.0.1:5000/predict?variance=2&skewness=3&curtosis=4&entropy=1