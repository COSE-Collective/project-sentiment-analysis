import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras_preprocessing.sequence import pad_sequences
from tqdm import tqdm
import numpy as np
from keras import optimizers
import zipfile
import io


class BiLSTM:
    def __init__(self, data, num_words=10000):
        self.Y_val, self.X_val = None, None
        self.X_train, self.X_test, self.Y_train, self.Y_test = data
        self.num_words = num_words

    def Train(self, epochs=20, batch_size=128, early_stop=True, patience=2, saving=False, path="results"):
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=0.3,
                                                                              random_state=42)
        print("TRAIN:" + str(len(self.X_train)), "VALIDATION:" + str(len(self.X_val)), "TEST:" + str(len(self.X_test)))
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=self.num_words)
        tokenizer.fit_on_texts(self.X_train)

        self.X_train = tokenizer.texts_to_sequences(self.X_train)
        self.X_val = tokenizer.texts_to_sequences(self.X_val)
        self.X_test = tokenizer.texts_to_sequences(self.X_test)
        self.X_train = pad_sequences(self.X_train)
        max_len = self.X_train.shape[1]
        self.X_test = pad_sequences(self.X_test, maxlen=max_len)
        self.X_val = pad_sequences(self.X_val, maxlen=max_len)

        print('GloVec embedding matrix building')
        embedding_vector = {}
        archive = zipfile.ZipFile('glove.42B.300d.zip', 'r')
        vocab_size = len(tokenizer.word_index) + 1
        f = io.TextIOWrapper(archive.open('glove.42B.300d.txt'), encoding="utf-8")
        for line in tqdm(f):
            value = line.split(' ')
            word = value[0]
            coef = np.array(value[1:], dtype='float32')
            embedding_vector[word] = coef
        embedding_matrix = np.zeros((vocab_size, 300))
        for word, i in tqdm(tokenizer.word_index.items()):
            embedding_value = embedding_vector.get(word)
            if embedding_value is not None:
                embedding_matrix[i] = embedding_value

        print('BiLSTM model building')
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, 300, input_length=self.X_train.shape[1], weights=[embedding_matrix],
                                      trainable=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.2)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        model.summary()
        optimizer = optimizers.Adam(learning_rate=0.005)
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer=optimizer,
                      metrics=['accuracy'])
        if early_stop:
            print('BiLSTM model training with EarlyStopping callback')
            callback = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=False,
                                                        monitor="val_loss")
            history = model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(self.X_val, self.Y_val), callbacks=[callback])
        else:
            print('BiLSTM model training')
            history = model.fit(self.X_train, self.Y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=(self.X_val, self.Y_val))
        print('Model successfully trained')
        if saving:
            print('Model saving')
            model.save(path + '/BiLSTM/BiLSTM.h5')
            print('Model saved')
        return model, history, self.X_test
