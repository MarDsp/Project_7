from flask import Flask, request
#import pandas as pd
#import numpy as np
import gc
import pickle
import sqlite3
import flask



file_directory = 'C:/Users/amaur/Documents/Projects/P_7_OKR/'
s_directory = 'C:/Users/amaur/Documents/Projects/P_7_OKR/site/'

with open(file_directory +'model.pkl', 'rb') as f:
    model= pickle.load(f)
with open(file_directory +'test_sample.pkl', 'rb') as f:
    test_s= pickle.load(f)


app = Flask(__name__)
@app.route('/')
def welcome():
    return flask.render_template('home-page.html')
 
@app.route('/predict', methods=['POST','GET'])
def predict():
    if flask.request.method == 'GET':
        return "Prediction page"
    if flask.request.method == 'POST':
        sk_id_curr=request.form.to_dict()
        #sk_id_curr =request.form.get('id')
        print("#######################")
        print(sk_id_curr)
        sk_id_curr = int(sk_id_curr)
        print("#######################")
        print(sk_id_curr)
        sk_id_curr = int(sk_id_curr)
        test_point=test_s.loc(sk_id_curr)
        test_point=test_s.loc[test_s['SK_ID_CURR'] == sk_id_curr]
        predicted_class = model.predict(test_point)
        predicted_proba = model.predict_proba(test_point)
        if predicted_class == 1:
            prediction = 'Non Solvable'
        else:
            prediction = 'Solvable'
                
        predicted_proba = 1 - predicted_proba
            
           
        return flask.render_template('result.html', output_proba = predicted_proba, output_class = prediction, sk_id_curr = sk_id_curr)
           
       
if __name__ == "__main__":
    app.run()




