import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import time
import pickle
from urllib.request import urlopen
import json
import requests as r

from explainer import interpretabilite



#file_directory = 'C:/Users/mascv/Documents/P_7_OKR/Dash/'
with open('test_sample.pkl', 'rb') as f:
    df= pickle.load(f)
with open('model.pkl', 'rb') as f:
    mod= pickle.load(f)
API_url_pred = "https://mardsp.pythonanywhere.com/predict"
list_id=list(df['SK_ID_CURR'].values)


st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")

#pred page
#if st.button("Veuillez saisir l\'identifiant d\'un client"):
id_input = st.number_input('Veuillez saisir l\'identifiant d\'un client:',None,None,100001,1 )
df_for_client=df.loc[df['SK_ID_CURR'] == id_input]
target=df_for_client['TARGET'].values
df_for_client=df_for_client.drop(['TARGET','SK_ID_CURR'],axis=1)
if int(id_input) in list_id:
    with st.spinner('Chargement du score du client...'):
        json_data = r.post(url= API_url_pred,json={'id':int(id_input)}).json()
        st.markdown("Caractéristiques de client:")
        st.dataframe(df_for_client)
        st.dataframe(df)
        info=json.loads(json_data)
        proba = info['output_proba']
        classe_predite = info['output_class']  
        chaine = 'Prédiction : '  +str(classe_predite)+  ' avec **' + str(round(proba*100)) + '%** de risque de défaut (classe réelle : '+str(target) + ')'

    st.markdown(chaine)
    st.subheader("Caractéristiques influençant le score")

    if st.button("Explain Results"):
       
        with st.spinner('Chargement des détails de la prédiction...'):
           
            exp=interpretabilite(df,df_for_client,mod)
            components.html(exp.as_html(), height=800)
else:
    st.write("Cet ID n'est pas encore dans la base de données, veuillez entrer un autre ID.")




