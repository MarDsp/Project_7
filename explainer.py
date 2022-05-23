def interpretabilite(dataframe,X,model):
    try:
        little_data=dataframe.drop(['TARGET','SK_ID_CURR'],axis=1)
    except:
        little_data=dataframe
    try:
        X=X.iloc[0,:]
    except:
        X
    lime1 = LimeTabularExplainer(little_data,
                                 feature_names=little_data.columns,
                                 class_names=["Solvable","Non Solvable"],
                                 discretize_continuous=False)                      
    exp = lime1.explain_instance(X,
                                 model.predict_proba,
                                 num_samples=200)
    return exp
