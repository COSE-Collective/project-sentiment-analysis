import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('stopwords')

import numpy as np
import pandas as pd
import emoji
from emoji import EMOJI_DATA
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pywsd.utils import lemmatize_sentence
import string
import re
import nltk

    
def remove_stopwords(i):
    stop_words = stopwords.words('english')
    filtered_sentence = []
    for w in i:
        if w not in stop_words:
            filtered_sentence.append(w)
    return filtered_sentence
def remove_emoji(i):
    for token in i:
        if token in EMOJI_DATA:
            i = re.sub(r'('+token+')', " ".join(emoji.demojize(token).replace(",","").replace(":","").replace("_"," ").split()), i)
    return i
def data_preprocessing(data):
    step=1
    print('Step {}/10: Renaming columns and removing "irrelevant" sentiments'.format(step))
    data=data.drop([0,1],axis=1)
    data=data.rename(columns={2:'Sentiment',3:'Tweet'})
    data.dropna(inplace=True)
    data=data[data['Sentiment']!='Irrelevant']
    data=data.drop_duplicates(keep = 'first')
    step+=1
    print('Step {}/10: Converting categorical data into integer format'.format(step))
    data["Sentiment"] = np.array(list(map(lambda y: 2 if y=="Positive" 
                                            else (1 if y=="Neutral" else 0),data["Sentiment"])))
    step+=1
    print('Step {}/10: Converting to lower case'.format(step))
    data['Tweet'] = [i.lower() for i in data['Tweet']]
    step+=1
    print('Step {}/10: Removing punctuation'.format(step))
    data['Tweet'] = [i.translate(str.maketrans('', '', string.punctuation)) for i in data['Tweet']]
    step+=1
    print('Step {}/10: Removing single words'.format(step))
    data['Tweet'] =  [re.sub(r"\b[a-zA-Z]\b", "", i) for i in data['Tweet']] 
    step+=1
    print('Step {}/10: Removing mentions and hastag symbols'.format(step))
    data['Tweet'] =  [re.sub(r"@[a-zA-Z0-9]+|\#[a-zA-Z0-9]","",str(i)) for i in data['Tweet']] 
    step+=1
    print('Step {}/10: Removing URLs'.format(step))
    data['Tweet'] =  [re.sub(r'http\S+', "", str(i)) for i in data['Tweet']] 
    data['Tweet'] =  [re.sub(r"[0-9]","",i) for i in data['Tweet']] 
    step+=1
    print('Step {}/10: Converting emoji to words'.format(step))
    data['Tweet']= [remove_emoji(i) for i in data['Tweet']]
    step+=1
    print('Step {}/10: Lemmatization'.format(step))
    data['Tweet'] =  [lemmatize_sentence(i) for i in data['Tweet']]
    step+=1
    print('Step {}/10: Removing stop words'.format(step))
    data['Tweet']= [remove_stopwords(i) for i in data['Tweet']]

    return data


if __name__ == '__main__':
    print('Data preprocessing')
    train_data=pd.read_csv('../../datasets/twitter_training.csv',header=None)
    val_data=pd.read_csv('../../datasets/twitter_test.csv',header=None)
    train_data=data_preprocessing(train_data)
    val_data=data_preprocessing(val_data)
    print('Data saving')
    train_data.to_csv('../../datasets/prepared_train.csv', encoding='utf-8', index=False)
    val_data.to_csv('../../datasets/prepared_test.csv', encoding='utf-8', index=False)
    print('Completed')

    
