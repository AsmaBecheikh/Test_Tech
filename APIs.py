# POST APIs

    # Api qui prend en entrée le json et qui retourne response = True si le json est correct
    # Api qui prend en entrée le json et qui retourne le max d'informations sur le modèle stocké (RMSE, R_square,...)
    # Api qui prend en entrée un json de test (même format que le json de train) et qui retourne la prédiction de la base de test   
#Import packages

from fastapi import FastAPI
from deserializer import check_id, check_var, check_json, deserialize_json
from model_training import clean_trainData, train_model, get_parameters, save_model, select_features
import json
from applying_model import get_date, prepare_testData, apply_latestStatModel
from math import sqrt


# Read the train data
f = open("train_input_cv.json")
data_train = json.load(f)

# read the test data
f1 = open("test_input_cv.json")
data_test = json.load(f1)

# Path of folder containing the .sav files
path = "AsmaBecheikh/Test_Technique_Asma"


appCheckJson = FastAPI()
appModel = FastAPI()
appPredictions = FastAPI()

@appCheckJson.get("/")
async def root():
        return {"message": check_json(data_train)}
    
@appModel.get("/")
async def root():
    df_train = deserialize_json(data_train)
    X_train, Y_train, X_test, Y_test, selected_features = clean_trainData(df_train, 9) #best model with most 9 correlated features with 
    model = train_model(X_train, Y_train)                                                #target
    save_model(model)
    tree_test_mse, tree_test_mae, tree_test_rmse = get_parameters(X_test, Y_test, model)
    Output = "Decision Tree test mse = ", tree_test_mse, " & mae = ", tree_test_mae, " & rmse = ", tree_test_rmse
    return {"Results": Output }

@appPredictions.get("/")
async def root():
    df_train = deserialize_json(data_train)
    notused1, notused2, notused3, notused4, selected_features = clean_trainData(df_train, 9)
    X_test, Y_test, min_max_scaler = prepare_testData(data_test, selected_features)
    y_pred = apply_latestStatModel(path, X_test)
    # y_pred=min_max_scaler.inverse_transform(y_pred)
    return {"predictions": y_pred.tolist() }

