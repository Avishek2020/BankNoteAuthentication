from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

# https://github.com/flasgger/flasgger
app = Flask(__name__)
Swagger(app)

pickle_in = open('/home/avishek/Jupyter_Notebooks/BankNoteAuthentication/classifier.pkl','rb')
classifier = pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Hello world"


@app.route('/predict',methods=["GET"])
def predict_note_authentication():    
    """Let Authenticate Bank Note
       Specifications
    ---
    parameters:        
       - name: variance
         in: query
         type: number
         required: true
       - name: skewness
         in: query
         type: number
         required: true 
       - name: curtosis
         in: query
         type: number
         required: true 
       - name: entropy
         in: query
         type: number
         required: true 
    responses:
        200:
            description: The output values
    """  
         
    variance = request.args.get('variance')
    skewness = request.args.get('skewness')
    curtosis = request.args.get('curtosis')
    entropy  = request.args.get('entropy')
    prediction = classifier.predict([[variance,skewness, curtosis,entropy]])
    return "The predicted Value is "+ str(prediction)



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
    """Let Authenticate Bank Note
       Specifications
    ---
    parameters:        
       - name: file
         in: formData
         type: file
         required: true       
    responses:
        200:
            description: The output values
    """  
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return "The predicted Values for the csv is "+ str(list(prediction))


   
    
if __name__=='__main__':
    app.run('127.0.0.1')
    

#http://127.0.0.1:5000/predict?variance=2&skewness=3&curtosis=4&entropy=1