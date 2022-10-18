import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import pickle
def LoadDataset():
    train_data1 = pd.read_csv('datasets/train1.csv')
    train_data2 = pd.read_csv('datasets/train2.csv')
    train_data=pd.concat([train_data1,train_data2])
    X_train, Y_train = train_data['Tweet'], train_data['Sentiment']
    test_data1 = pd.read_csv('datasets/test1.csv')
    test_data2 = pd.read_csv('datasets/test2.csv')
    test_data=pd.concat([test_data1,test_data2])
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