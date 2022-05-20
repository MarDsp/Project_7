
import streamlit as st
import numpy as np
import pandas as pd
import time
import pickle
from urllib.request import urlopen
import json
import requests as r


file_directory = 'C:/Users/amaur/Documents/Projects/P_7_OKR/site/'
with open(file_directory +'test_sample.pkl', 'rb') as f:
    df= pickle.load(f)
id_input = st.text_input('Veuillez saisir l\'identifiant d\'un client:', )
API_url= "https://mardsp.pythonanywhere.com"
API_url_pred = "https://mardsp.pythonanywhere.com/predict"

list_id=list(df['SK_ID_CURR'].values)
#page d'accueil
st.dataframe(df)
#pred page
if int(id_input) in list_id:
    with st.spinner('Chargement du score du client...'):
        json_data = r.post(url= API_url_pred,json={'id':int(id_input)}).json()
        st.dataframe(json_data)
        classe_predite = json_data['output_class']        
        proba = json_data['output_proba']
        chaine = 'Prédiction : **'  +  '** avec **' + str(round(proba*100)) + '%** de risque de défaut (classe réelle : '+str(classe_predite) + ')'

    st.markdown(chaine)

    st.subheader("Caractéristiques influençant le score")






