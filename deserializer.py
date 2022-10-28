
# QUESTIONS : 
    # Développement d'un service capable de vérifier si le json envoyé par l'utilisateur est correct ou pas PUIS le convertir en dataFrame.
    # Transformation de ce service en API (port 6000)

# NB : 
    # Vous pouvez ajouter d'autres fonctions si vous juger cela nécessaire. 
    # NE PAS METTRE DES FONCTIONS HORS LE CONTEXTE DE CONVERSION DU JSON ICI.

# INDICATIONS : 
    # Checker si tous les objets du json ont bien un ID
    # Checker si tous les sousèobjets du json ont bien le même nombre de variables
    # ...

# ATTENTION : Les 2 fonctions que j'ai listé ici doivent être présentes dans votre code sous les même noms

# import packages 
import pandas as pd

def check_id(x):
    """Check if all objects of the json file have Ids"""
    s = 0
    if 'id' not in x.keys():
        s = 1
    return(s)

def check_var(x, ref):
    """Check if all subobjects of the json file have the same number of variables"""
    s = 0
    if len(x) != len(ref):
        s = 1
    return(s)


def check_json(data):
    """Check if the sent json is correct"""
    
    ref_infos = data['data_train_assets'][0]['asset_infos']
    ref_scores = data['data_train_assets'][0]['asset_scores']
    s_infos = 0
    s_scores = 0
    
    s_ids = sum(list(map(check_id, data['data_train_assets'])))

    for i in range(len(data['data_train_assets'])):
        s_infos += check_var(data['data_train_assets'][i]['asset_infos'], ref_infos)
        s_scores += check_var(data['data_train_assets'][i]['asset_scores'], ref_scores)
    if s_ids == 0 and s_infos == 0 and s_scores == 0:
        return (True)
    else:
        return(False)
    pass

def deserialize_json(data):
    """Deserialize the input json to build the dataFrame to put in the statistical model"""
    
    df = pd.DataFrame(data['data_train_assets'])
    df_infos = df['asset_infos'].apply(pd.Series)
    df_scores = df['asset_scores'].apply(pd.Series)
    df_all = pd.concat([df.drop(['asset_infos', 'asset_scores'], axis=1), df_infos, df_scores], axis=1)
    
    return(df_all)

    pass
