import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import pickle
def LoadDataset():
    train_data = pd.read_csv('datasets/prepared_train.csv')
    X_train, Y_train = train_data['Tweet'], train_data['Sentiment']
    test_data = pd.read_csv('datasets/prepared_test.csv')
    X_test, Y_test = test_data['Tweet'], test_data['Sentiment']
    data=pd.concat([train_data,test_data])
    return X_train,X_test,Y_train,Y_test, data


def LoadModel(modelName):
    loaded_model = None
    if modelName == 'RF':
        loaded_model = pickle.load(open('src/Models/RandomForestClassifier/RandomForestClassifier.pkl', 'rb'))
    else :
        loaded_model = load_model('src/Models/'+ modelName +'/'+ modelName +'.h5')
    return loaded_model