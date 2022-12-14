import pandas as pd
from transformers import TFBertModel, TFRobertaModel
import tensorflow as tf
from sklearn.utils import shuffle


def dataset_loading():
    train_data1 = pd.read_csv('datasets/prepared_train1.csv')
    train_data1 = shuffle(train_data1)
    train_data2 = pd.read_csv('datasets/prepared_train2.csv')
    train_data2 = shuffle(train_data2)
    train_data = pd.concat([train_data1, train_data2])
    train_data = shuffle(train_data)
    train_data = train_data[(train_data['Tweet'].str.len() < 512)]
    X_train, Y_train = train_data['Tweet'], train_data['Sentiment']

    test_data1 = pd.read_csv('datasets/prepared_test1.csv')
    test_data2 = pd.read_csv('datasets/prepared_test2.csv')
    test_data3 = pd.read_csv('datasets/prepared_test3.csv')
    test_data = pd.concat([test_data1, test_data2])
    test_data = pd.concat([test_data, test_data3])
    test_data = test_data[(test_data['Tweet'].str.len() < 512)]

    # test_data = test_data.head(200)
    X_test, Y_test = test_data['Tweet'], test_data['Sentiment']

    data = pd.concat([train_data, test_data])
    return X_train, X_test, Y_train, Y_test, data


def model_loading(model_name):
    model = None
    if model_name == "LSTM" or model_name == "BiLSTM":
        model = tf.keras.models.load_model('src/Models/' + model_name + '/' + model_name + '.h5')
    elif model_name == "BERT":
        model = tf.keras.models.load_model('src/Models/' + model_name + '/' + model_name + '.h5',
                                           custom_objects={'TFBertModel': TFBertModel})
    elif model_name == "RoBERTa":
        model = tf.keras.models.load_model('src/Models/' + model_name + '/' + model_name + '.h5',
                                           custom_objects={'TFRobertaModel': TFRobertaModel})
    return model
