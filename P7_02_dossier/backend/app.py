from flask import Flask, jsonify
import pandas as pd
import pickle
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

PATH = 'data/'
#Chargement des données 

df = pd.read_parquet(PATH+'test_df.parquet')
#df = pd.read_csv(PATH+'test_df.csv')
print('df shape = ', df.shape)

#Chargement du modèle
model = pickle.load(open('./LGBMClassifier.pkl', 'rb'))


@app.route('/')
def hello():
    return 'Hello, World!'

@app.route('/credit/<id_client>')
def credit(id_client):

    print('id client = ', id_client)
    
    #Récupération des données du client en question
    ID = int(id_client)
    X = df[df['SK_ID_CURR'] == ID]
    
    ignore_features = ['Unnamed: 0','SK_ID_CURR', 'INDEX', 'TARGET']
    relevant_features = [col for col in df.columns if col not in ignore_features]

    X = X[relevant_features]
    
    print('X shape = ', X.shape)
    
    proba = model.predict_proba(X)
    prediction = model.predict(X)

    #DEBUG
    #print('id_client : ', id_client)
  
    dict_final = {
        'prediction' : int(prediction),
        'proba' : float(proba[0][0])
        }

    print('Nouvelle Prédiction : \n', dict_final)

    return jsonify(dict_final)




#lancement de l'application
if __name__ == "__main__":
    app.run(debug=True)
    
    