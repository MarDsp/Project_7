
import streamlit as st
import numpy as np
import pandas as pd
import time
import pickle
from urllib.request import urlopen
import json
import requests as r
import lime
import lime.lime_tabular
from lime.lime_tabular import LimeTabularExplainer



file_directory = 'C:/Users/amaur/Documents/Projects/P_7_OKR/site/'
with open(file_directory +'test_sample.pkl', 'rb') as f:
    df= pickle.load(f)
API_url_pred = "https://mardsp.pythonanywhere.com/predict"
list_id=list(df['SK_ID_CURR'].values)


st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")

id_input = st.number_input('Veuillez saisir l\'identifiant d\'un client:',None,None,100001,1 )


df_for_client=df.loc[df['SK_ID_CURR'] == id_input]
target=df_for_client['TARGET'].values
df_for_client=df_for_client.drop(['TARGET'],axis=1)

#pred page

if int(id_input) in list_id:
    with st.spinner('Chargement du score du client...'):
        json_data = r.post(url= API_url_pred,json={'id':int(id_input)}).json()
        st.dataframe(df_for_client)
        info=json.loads(json_data)

              
        proba = (info['output_proba'])
        classe_predite = info['output_class']  
        chaine = 'Prédiction : '  +str(classe_predite)+  ' avec **' + str(round(proba*100)) + '%** de risque de défaut (classe réelle : '+str(target) + ')'

    st.markdown(chaine)

    st.subheader("Caractéristiques influençant le score")
else:
    st.write("Cet ID n'est pas encore dans la base de données, veuillez entrer un autre ID.")
    

def interpretabilite():
    lime1 = LimeTabularExplainer(df_for_client,feature_names=df_for_client.columns, class_names=["Solvable", "Non Solvable"],discretize_continuous=False)                      
    exp = lime1.explain_instance(df_for_client.iloc[100],model.predict_proba, num_samples=100)


