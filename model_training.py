# QUESTIONS : 
    # Développement d'un service capable d'entrainer un modèle statistique de prédiction sur la base de train (les informations nécessaires doivent être comme des paramètres des fonnctions).
    # Transformation de ce service en API (port 6000)

# NB : 
    # Vous pouvez ajouter d'autres fonctions si vous juger cela nécessaire. 
    # NE PAS METTRE DES FONCTIONS HORS LE CONTEXTE DE LA PREDICTION.

# ATTENTION : Les 4 fonctions que j'ai listé ici doivent être présentes dans votre code sous les même noms

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


def select_features(X, Y, K):
    """Function to select features with pearson correlation"""
    new_features = []
    fs = SelectKBest(score_func=f_regression, k=K)
    X_selected = fs.fit_transform(X, Y)
    mask = fs.get_support()
    feature_names = list(X.columns.values)
    for bool, feature in zip(mask, feature_names):
        if bool:
            new_features.append(feature)
    X_selected = pd.DataFrame(X_selected, columns=new_features)
    return(X_selected)
    pass
    
    
def clean_trainData(df_train, K):
    """Cleaning, splitting the training data and dummies the cateogial variables"""
    
    df = df_train.drop(['id'], axis=1)
    
    dataframe_dummies = pd.get_dummies(df, columns=['var1', 'var4', 'var6', 'var8', 'var9'])
    columns = list(dataframe_dummies.columns.values)
    
    min_max_scaler = preprocessing.MinMaxScaler()
    dataframe_scaled = min_max_scaler.fit_transform(dataframe_dummies)
    dataframe = pd.DataFrame(dataframe_scaled, columns=columns)
    
    X = dataframe.iloc[:, 1:len(dataframe)]
    Y = dataframe['target']
    X_selected = select_features(X, Y, K)
    X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.2, random_state=123)
    selected_features = X_train.columns.tolist()
    return(X_train, Y_train, X_test, Y_test, selected_features)
    
    pass

def train_model(X_train, Y_train):
    """Apply the model to the train data and the train target by turning multiple parameters"""
    
    model = DecisionTreeRegressor(max_depth=9, max_features=None, max_leaf_nodes=60, min_samples_leaf=3, 
                                             min_weight_fraction_leaf=0.1, splitter='random')
    model.fit(X_train, Y_train)
    
    return(model)
    pass

def get_parameters(X_test, Y_test, model):
    """Calculate statistical parameters of the model (EX : RMSE)"""
    
    tree_test_mse = mean_squared_error(Y_test, model.predict(X_test))
    tree_test_mae = mean_absolute_error(Y_test, model.predict(X_test))
    tree_test_rmse = sqrt(tree_test_mse)
    
    return(tree_test_mse, tree_test_mae, tree_test_rmse)
    pass

def save_model(model):
    """Save the statistical model in a file .sav"""
    
    filename = 'Decision_tree_model_'+ time.strftime("%Y_%m_%d_%H_%M_%S") + '.sav'
    pickle.dump(model, open(filename, 'wb'))
    
    pass