import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from keras.initializers import Constant
from tqdm import tqdm
import nltk
import numpy as np
from keras import optimizers
import zipfile
import io
class BiLSTM:
    def __init__(self, data):
        self.X_train,  self.X_test, self.Y_train, self.Y_test= data

    def Train(self):
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=0.3, random_state = 42)

        num_words = 10000
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(self.X_train)

        self.X_train = tokenizer.texts_to_sequences(self.X_train)
        self.X_val = tokenizer.texts_to_sequences(self.X_val)
        self.X_test = tokenizer.texts_to_sequences(self.X_test)

        vocab_size = len(tokenizer.word_index) + 1

        self.X_train = pad_sequences(self.X_train)
        max_len = self.X_train.shape[1]
        self.X_test = pad_sequences(self.X_test, maxlen = max_len)
        self.X_val = pad_sequences(self.X_val, maxlen = max_len)
        print('GloVec embedding matrix building')
        embedding_vector = {}
        archive = zipfile.ZipFile('glove.42B.300d.zip', 'r')
        #f = archive.open('glove.42B.300d.txt')
        
        #f = open('glove.42B.300d.txt')
        vocab_size = len(tokenizer.word_index)+1
        f= io.TextIOWrapper(archive.open('glove.42B.300d.txt'), encoding="utf-8") 
        for line in tqdm(f):
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:],dtype = 'float32')
            embedding_vector[word] = coef
        embedding_matrix = np.zeros((vocab_size,300))
        for word,i in tqdm(tokenizer.word_index.items()):
            embedding_value = embedding_vector.get(word)
            if embedding_value is not None:
                embedding_matrix[i] = embedding_value
        embedding_matrix.shape

        n_dim = 300
        lstm_out = 64
        print('BiLSTM model training')
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, n_dim, input_length = self.X_train.shape[1],weights = [embedding_matrix ] , trainable = True),
            tf.keras.layers.SpatialDropout1D(0.2),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_out, dropout = 0.3, recurrent_dropout = 0.2)),
            tf.keras.layers.Dense(3,activation = 'softmax')
        ])
        model.summary()
        optimizer = optimizers.Adam(learning_rate=0.005)
        model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer = optimizer, metrics = ['accuracy'])
        batch_size = 128
        epochs = 90
        history = model.fit(self.X_train, self.Y_train, epochs = epochs,batch_size = batch_size, validation_data = (self.X_val, self.Y_val))
#         epochs = 20
#         callback = tf.keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = False, monitor = "val_loss")
#         history = model.fit(self.X_train, self.Y_train, epochs = epochs,batch_size = batch_size, validation_data = (self.X_val, self.Y_val),callbacks = [callback])
        print('Model built successfully')
        print('Model saving')
        model.save('src/Models/BiLSTM/BiLSTM.h5')
        print('Model saved')
        return history, self.X_test, epochs

