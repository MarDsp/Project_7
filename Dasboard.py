
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
id_input = st.number_input('Veuillez saisir l\'identifiant d\'un client:',None,None,100001,1 )
API_url= "https://mardsp.pythonanywhere.com"
API_url_pred = "https://mardsp.pythonanywhere.com/predict"




#@st.cache #mise en cache de la fonction pour exécution unique
#def chargement_explanation(id_input, dataframe, model, sample):
    #return interpretation(str(id_input), 
        #dataframe, 
        #model, 
        #sample=sample)

list_id=list(df['SK_ID_CURR'].values)
#page d'accueil

df_for_client=df.loc[df['SK_ID_CURR'] == id_input]
#pred page
st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")
if int(id_input) in list_id:
    with st.spinner('Chargement du score du client...'):
        json_data = r.post(url= API_url_pred,json={'id':int(id_input)}).json()
        st.dataframe(df_for_client)
        info=json.loads(json_data)
              
        proba = (info['output_proba'])
        classe_predite = info['output_class']  
        chaine = 'Prédiction : ** '  +str(classe_predite)+  '** avec **' + str(round(proba*100)) + '%** de risque de défaut (classe réelle : '+str(df_for_client['TARGET'].values) + ')'

    st.markdown(chaine)

    st.subheader("Caractéristiques influençant le score")






