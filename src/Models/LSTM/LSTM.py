import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from keras.initializers import Constant
import numpy as np
from keras import optimizers
class LSTM:
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
        self.X_test = pad_sequences(self.X_test, maxlen=max_len)
        self.X_val = pad_sequences(self.X_val, maxlen=max_len)

        n_dim = 128
        lstm_out = 128
        print('LSTM model training')
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, n_dim, input_length =max_len),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.summary()
        optimizer = optimizers.Adam(learning_rate=0.005)
        model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False), optimizer = optimizer, metrics = ['accuracy'])
        batch_size = 128
#         epochs = 90
#         history = model.fit(self.X_train, self.Y_train, epochs = epochs,batch_size = batch_size, validation_data = (self.X_val, self.Y_val))
        epochs = 20
        callback = tf.keras.callbacks.EarlyStopping(patience = 2, restore_best_weights = False, monitor = "val_loss")
        history = model.fit(self.X_train, self.Y_train, epochs = epochs,batch_size = batch_size, validation_data = (self.X_val, self.Y_val),callbacks = [callback])
        print('Model built successfully')
        print('Model saving')
        model.save('src/Models/LSTM/LSTM.h5')
        print('Model saved')
        return history, self.X_test, epochs

