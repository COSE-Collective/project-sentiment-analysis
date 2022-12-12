import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from transformers import TFBertModel, BertTokenizerFast


def tokenize(data, max_len, tokenizer):
    input_ids = []
    attention_masks = []

    aleqmsg = 'all inputs have the same length'

    print("\n\nELEMENT LENGTH")
    for i in range(len(data)):
        encoded = tokenizer.encode_plus(
            data[i],
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            return_attention_mask=True
        )
        input_ids.append(encoded['input_ids'])
        #print(np.shape(input_ids))
        attention_masks.append(encoded['attention_mask'])

        if len(encoded['input_ids']) != 512:
            aleqmsg = 'some inputs have a different length'
            print("element " + str(i) + " has a different length: " + str(len(encoded['input_ids'])))

    print(aleqmsg)

        #if i >= len(data) - 10:
        #    pepe = np.array(input_ids, dtype=object)
        #    print(np.shape(input_ids))
        #    print(np.shape(pepe))

    try:
        a = np.array(input_ids).reshape(len(data), 512)
    except:
        a = np.array(input_ids)

    #print("\n\nInput_ids")
    #print(input_ids[0])

    b = np.array(attention_masks)
    print("\n\nShape input_ids")
    print(np.shape(a))
    print("\nShape attention_masks")
    print(np.shape(b))
    print("\n\n")

    return np.array(input_ids), np.array(attention_masks)


def create_model(bert_layer, max_len):
    print('BERT model building')
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


class BERT:
    def __init__(self, data, max_len=512):
        self.X_val, self.Y_val = None, None
        self.X_train, self.X_test, self.Y_train, self.Y_test = data
        self.max_len = max_len

    def Train(self, epochs=9, batch_size=8, early_stop=True, patience=1, saving=False):
        self.X_train, self.X_val, self.Y_train, self.Y_val = train_test_split(self.X_train, self.Y_train, test_size=0.3,
                                                                              random_state=42)
        self.X_train.reset_index(drop=True, inplace=True)
        self.X_test.reset_index(drop=True, inplace=True)
        self.X_val.reset_index(drop=True, inplace=True)
        print("TRAIN:" + str(len(self.X_train)), "VALIDATION:" + str(len(self.X_val)), "TEST:" + str(len(self.X_test)))
        tokenizer_bert = BertTokenizerFast.from_pretrained('bert-base-uncased')

        train_input_ids, train_attention_masks = tokenize(self.X_train, self.max_len, tokenizer_bert)
        val_input_ids, val_attention_masks = tokenize(self.X_val, self.max_len, tokenizer_bert)
        test_input_ids, test_attention_masks = tokenize(self.X_test, self.max_len, tokenizer_bert)

        bert_base_uncased = TFBertModel.from_pretrained('bert-base-uncased')
        bert_model = create_model(bert_base_uncased, self.max_len)
        bert_model.summary()
        if early_stop:
            print('BERT model training with EarlyStopping callback')
            callback = tf.keras.callbacks.EarlyStopping(patience=patience, restore_best_weights=True,
                                                        monitor="val_loss")
            history_bert = bert_model.fit([train_input_ids, train_attention_masks], self.Y_train,
                                          validation_data=([val_input_ids, val_attention_masks], self.Y_val),
                                          epochs=epochs,
                                          batch_size=batch_size, callbacks=[callback])
        else:
            print('BERT model training')
            history_bert = bert_model.fit([train_input_ids, train_attention_masks], self.Y_train,
                                          validation_data=([val_input_ids, val_attention_masks], self.Y_val),
                                          epochs=epochs,
                                          batch_size=batch_size)
        print('Model successfully trained')
        if saving:
            print('Model saving')
            bert_model.save('src/Models/BERT/BERT.h5')
            print('Model saved')
        return bert_model, history_bert, [test_input_ids, test_attention_masks]
