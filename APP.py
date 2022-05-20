from flask import Flask, request
#import pandas as pd
#import numpy as np
import gc
import pickle

import flask
import json
from flask import Flask, render_template, jsonify, request, flash, redirect, url_for



file_directory = 'C:/Users/amaur/Documents/Projects/P_7_OKR/site/'


with open(file_directory +'model.pkl', 'rb') as f:
    model= pickle.load(f)
with open(file_directory +'test_sample.pkl', 'rb') as f:
    test_s= pickle.load(f)


app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def welcome():
    
    if flask.request.method == 'POST':
        return(redirect(url_for('predict'), code=307)) 
    else:
        return ('Veuillez compléter le champ. ')
    
 
@app.route('/predict', methods=['POST','GET'])
def predict():
    if flask.request.method == 'POST':
        sk_id_curr=request.get_json()["id"]
        sk_id_curr = int(sk_id_curr)
        test_point=test_s.loc[test_s['SK_ID_CURR'] == sk_id_curr]
        predicted_class = model.predict(test_point)
        predicted_proba = float(model.predict_proba(test_point)[:, 1])
        if predicted_class == 1:
            prediction = 'Non Solvable'
        else:
            prediction = 'Solvable'
        predicted_proba = 1 - predicted_proba
        output={'output_proba' : predicted_proba,'output_class' : prediction,'sk_id_curr' : sk_id_curr} 
        output = json.dumps(output)  
        return jsonify( output)
           
       
if __name__ == "__main__":
    app.run(debug=True)




