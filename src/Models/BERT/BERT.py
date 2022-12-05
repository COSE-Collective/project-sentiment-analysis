import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import TFBertModel, BertTokenizerFast


class BERT:
    def __init__(self, data):
        self.X_val, self.Y_val = None, None
        self.X_train, self.X_test, self.Y_train, self.Y_test = data

    def Train(self):
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=0.3,
                                                                              random_state=42)
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.X_val.reset_index(drop=True, inplace=True)
        print(len(self.X_train), len(self.X_val), len(self.X_test))
        MAX_LEN = 512
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

        def tokenize(data, max_len=MAX_LEN):
            input_ids = []
            attention_masks = []
            for i in range(len(data)):
                encoded = tokenizer.encode_plus(
                    data[i],
                    add_special_tokens=True,
                    max_length=max_len,
                    padding='max_length',
                    return_attention_mask=True
                )
                input_ids.append(encoded['input_ids'])
                attention_masks.append(encoded['attention_mask'])
            return np.array(input_ids), np.array(attention_masks)

        def create_model(bert_layer, max_len=MAX_LEN):
            input_ids = tf.keras.Input(shape=(max_len,), dtype='int32')
            attention_masks = tf.keras.Input(shape=(max_len,), dtype='int32')
            embeddings = bert_layer([input_ids, attention_masks])[1]
            layer = tf.keras.layers.Dense(256, activation="relu")(embeddings)
            output = tf.keras.layers.Dense(3, activation="softmax")(layer)

            model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)

            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=5e-5, decay=1e-7)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
            accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

            model.compile(optimizer, loss=loss, metrics=accuracy)

            return model

        train_input_ids, train_attention_masks = tokenize(self.X_train)
        val_input_ids, val_attention_masks = tokenize(self.X_val)
        test_input_ids, test_attention_masks = tokenize(self.X_test)

        bert_base_uncased = TFBertModel.from_pretrained('bert-base-uncased')
        bert_model = create_model(bert_base_uncased, MAX_LEN)
        bert_model.summary()

        callback = tf.keras.callbacks.EarlyStopping(patience=1, restore_best_weights=True, monitor="val_loss")
        history_bert = bert_model.fit([train_input_ids, train_attention_masks], self.Y_train,
                                      validation_data=([val_input_ids, val_attention_masks], self.Y_val), epochs=9,
                                      batch_size=8, callbacks=[callback])
        print('Model built successfully')
        print('Model saving')
        bert_model.save('src/Models/BERT/BERT.h5')
        print('Model saved')
        return history_bert, [test_input_ids, test_attention_masks]
