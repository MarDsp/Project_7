import streamlit as st
import streamlit.components.v1 as components
from catboost import CatBoostClassifier
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




#file_directory = 'C:/Users/mascv/Documents/P_7_OKR/Dash/'
with open('test_sample.pkl', 'rb') as f:
    df= pickle.load(f)
with open('model.pkl', 'rb') as f:
    mod= pickle.load(f)
API_url_pred = "https://mardsp.pythonanywhere.com/predict"
list_id=list(df['SK_ID_CURR'].values)
var_imp=['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3']

st.title('Dashboard Scoring Credit')
st.subheader("Prédictions de scoring client et comparaison à l'ensemble des clients")

st.sidebar.title("Exemples d'ID client...")
for id in list_id:
    st.sidebar.info(id)
#pred page

if st.checkbox('Montrer dataframe'):
    st.dataframe(df)
id_input = st.number_input('Veuillez saisir l\'identifiant d\'un client:',None,None,100001,1 )
df_for_client=df.loc[df['SK_ID_CURR'] == id_input]
target=df_for_client['TARGET'].values
df_for_client=df_for_client.drop(['TARGET','SK_ID_CURR'],axis=1)
if int(id_input) in list_id:
    with st.spinner('Chargement du score du client...'):
        json_data = r.post(url= API_url_pred,json={'id':int(id_input)}).json()
        st.markdown("Caractéristiques de client:")
        st.dataframe(df_for_client)
        info=json.loads(json_data)
        proba = info['output_proba']
        classe_predite = info['output_class']  
        chaine = 'Prédiction : '  +str(classe_predite)+  ' avec probabilité :' + str(round(proba*100)) + '%'

    st.markdown(chaine)
    st.subheader("Caractéristiques influençant le score")

    if st.button("Explain Results"):
       
        with st.spinner('Chargement des détails de la prédiction...'):
           
            def interpretabilite(dataframe,X,model):
                little_data=dataframe.drop(['TARGET','SK_ID_CURR'],axis=1)
                X=X.iloc[0,:]
                lime1 = LimeTabularExplainer(little_data,
                                feature_names=little_data.columns, 
                                class_names=["Solvable","Non Solvable"],
                                discretize_continuous=False)                      
                exp = lime1.explain_instance(X,
                                model.predict_proba)
                return exp
            exp=interpretabilite(df,df_for_client,mod)
            components.html(exp.as_html(), height=800)
        #st.subheader("Les distribution des variables importantes")
    if st.checkbox('Les distribution des variables importantes'):
        selected_var = st.multiselect("Сhoisir des variable:", df.drop(['TARGET','SK_ID_CURR'],axis=1).columns.tolist(), default=var_imp)
        variables=selected_var
        for v in variables:
            f = px.histogram(df, x=v, nbins=15, title="Distribution de "+str(v))
            f.update_xaxes(title=str(v))
            f.update_yaxes(title="Nombre de classes")
            st.plotly_chart(f)
            st.write(f"Valeur de la variable {v!r} pour le client avec un ID '{id_input!r}' est: {df_for_client.iloc[0][v]!r}")

else:
    st.write("Cet ID n'est pas encore dans la base de données, veuillez entrer un autre ID.")




