# QUESTIONS : 
    # Développement d'un service capable d'appliquer un modèle entrainer à une base de test (les informations nécessaires doivent être comme des paramètres des fonnctions).
    # Transformation de ce service en API (port 6000)

# NB : 
    # Vous pouvez ajouter d'autres fonctions si vous juger cela nécessaire. 
    # NE PAS METTRE DES FONCTIONS HORS LE CONTEXTE DE L'APPLICATION DU MODELE'.

# ATTENTION : Les 2 fonctions que j'ai listé ici doivent être présentes dans votre code sous les même noms

from deserializer import deserialize_json
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import make_regression 
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from math import sqrt
from sklearn.tree import DecisionTreeRegressor
import time
import pickle
import re
import datetime
from os import walk

def get_date(filename):
    """function to get the last model .sav"""
    date_pattern = re.compile(r'(\d{4})_(\d{2})_(\d{2})_(\d{2})_(\d{2})_(\d{2})')
    matched = date_pattern.search(filename)
    if not matched:
        return None
    y, m, d, H, M, S = map(int, matched.groups())
    return datetime.datetime(y, m, d, H, M, S)

def prepare_testData(df_test, selected_features):
    """Cleaning, splitting the testing data and apply the same training dumification to this data."""
    test_data = deserialize_json(df_test)
    test = test_data.drop(['id'], axis=1)
    
    dataframe_dummies = pd.get_dummies(test, columns=['var1', 'var4', 'var6', 'var8', 'var9'])
    columns = list(dataframe_dummies.columns.values)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    dataframe_scaled = min_max_scaler.fit_transform(dataframe_dummies)
    dataframe = pd.DataFrame(dataframe_scaled, columns=columns)
    
    X_test = dataframe[selected_features]
    Y_test = dataframe['target']
    return(X_test, Y_test, min_max_scaler)
    pass

def apply_latestStatModel(path, X_test):
    """Application of the last statistical model saved to the test base"""
    
    filenames = next(walk(path), (None, None, []))[2]
    dates = (get_date(fn) for fn in filenames)
    dates = (d for d in dates if d is not None)
    last_date = max(dates)
    last_date = last_date.strftime('%Y_%m_%d_%H_%M_%S')
    filenames = [fn for fn in filenames if last_date in fn]
    for fn in filenames:
        loaded_model = pickle.load(open(fn, 'rb'))
    predictions = loaded_model.predict(X_test)
    return(predictions)
    
    pass