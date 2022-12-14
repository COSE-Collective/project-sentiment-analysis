import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import RobertaTokenizerFast, TFRobertaModel


def tokenize_roberta(data, max_len, tokenizer):
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


def create_model(bert_layer, max_len):
    print('RoBERTa model building')
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=2e-5)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')

    input_ids = tf.keras.Input(shape=(max_len,), dtype='int32')
    attention_masks = tf.keras.Input(shape=(max_len,), dtype='int32')
    embeddings = bert_layer([input_ids, attention_masks])[1]
    output = tf.keras.layers.Dense(3, activation="softmax")(embeddings)

    model = tf.keras.models.Model(inputs=[input_ids, attention_masks], outputs=output)
    model.compile(opt, loss=loss, metrics=accuracy)

    return model


class ROBERTA:
    def __init__(self, data, max_len=512, size=30000):
        self.X_val, self.Y_val = None, None
        self.X_train, self.X_test, self.Y_train, self.Y_test = data
        self.X_train, self.Y_train = self.X_train.head(size), self.Y_train.head(size)
        self.max_len = max_len

    def Train(self, epochs=5, batch_size=8, early_stop=False, patience=2, saving=False):
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=0.3,
                                                                              random_state=42)
        print(len(self.X_train), len(self.X_val), len(self.X_test))
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.X_val.reset_index(drop=True, inplace=True)

        tokenizer_roberta = RobertaTokenizerFast.from_pretrained("roberta-base")

        train_input_ids, train_attention_masks = tokenize_roberta(self.X_train, self.max_len, tokenizer_roberta)
        val_input_ids, val_attention_masks = tokenize_roberta(self.X_val, self.max_len, tokenizer_roberta)
        test_input_ids, test_attention_masks = tokenize_roberta(self.X_test, self.max_len, tokenizer_roberta)

        roberta_base = TFRobertaModel.from_pretrained('roberta-base')

        roberta_model = create_model(roberta_base, self.max_len)
        roberta_model.summary()
        if early_stop:
            print('RoBERTa model training with EarlyStopping callback')
            callback = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True,
                                                        monitor="val_loss")
            history_roberta = roberta_model.fit([train_input_ids, train_attention_masks], self.Y_train,
                                                validation_data=([val_input_ids, val_attention_masks], self.Y_val),
                                                epochs=epochs,
                                                batch_size=batch_size, callbacks=[callback])
        else:
            print('RoBERTa model training')
            history_roberta = roberta_model.fit([train_input_ids, train_attention_masks], self.Y_train,
                                                validation_data=([val_input_ids, val_attention_masks], self.Y_val),
                                                epochs=epochs,
                                                batch_size=batch_size)
        print('Model successfully trained')
        if saving:
            print('Model saving')
            roberta_model.save('src/Models/RoBERTa/RoBERTa.h5')
            print('Model saved')

        return roberta_model, history_roberta, [test_input_ids, test_attention_masks]
