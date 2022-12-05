import nltk
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt')
# nltk.download('stopwords')
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
    stop_words.remove('not')
    stop_words.remove('no')
    filtered_sentence = []
    for w in i:
        if w not in stop_words:
            filtered_sentence.append(w)
    #print(stop_words)
    return filtered_sentence
def remove_emoji(i):
    for token in i:
        if token in EMOJI_DATA:
            i = re.sub(r'('+token+')', " ".join(emoji.demojize(token).replace(",","").replace(":","").replace("_"," ").split()), i)
    return i
def func_not(text):
    contractions = {
    "ain't": "am not / are not",
    "ain`t": "am not / are not",
    "aint": "am not / are not",   
    "aren't": "are not / am not",
    "aren`t": "are not / am not",
    "arent": "are not / am not",
    "amnt": "am not", 
    "can't": "cannot",
    "can`t": "cannot",
    "cant": "cannot",
    "can't've": "cannot have",
    "can`t`ve": "cannot have",
    "'cause": "because",
    "could`ve": "could have",
    "could've": "could have",
    "couldn't": "could not",
    "couldn`t": "could not",
    "couldnt": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "didn`t": "did not",
    "didnt": "did not",
    "doesn't": "does not",
    "doesn`t": "does not",
    "doesnt": "does not",
    "don't": "do not",
    "don`t": "do not",
    "dont": "do not",
    "hadn't": "had not",
    "hadn`t": "had not",
    "hadnt": "had not",
    "hadn't've": "had not have",
    "hadn`t`ve": "had not have",
    "hasnt": "has not",
    "hasn't": "has not",
    "hasn`t": "has not",
    "haven't": "have not",
    "haven`t": "have not",
    "havent": "have not",
    "he'd": "he had / he would",
    "he'd've": "he would have",
    "he`d": "he had / he would",
    "he`d`ve": "he would have",
    "he'll": "he shall / he will",
    "he`ll": "he shall / he will",
    "he'll've": "he shall have / he will have",
    "he`ll`ve": "he shall have / he will have",
    "he's": "he has / he is",
    "he`s": "he has / he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how has / how is",
    "i'd": "I had / I would",
    "i'd've": "I would have",
    "i'll": "I shall / I will",
    "i'll've": "I shall have / I will have",
    "i`m": "I am",
    "i've": "I have",
    "isn't": "is not",
    "isnt": "is not",
    "how`d": "how did",
    "how`d`y": "how do you",
    "how`ll": "how will",
    "how`s": "how has / how is",
    "i`d": "I had / I would",
    "i`d`ve": "I would have",
    "i`ll": "I shall / I will",
    "i`ll`ve": "I shall have / I will have",
    "i'm": "I am",
    "i`ve": "I have",
    "isn`t": "is not",
    "it`d": "it had / it would",
    "it'd": "it had / it would",
    "it'd've": "it would have",
    "it`d`ve": "it would have",
    "it`ll": "it shall / it will",
    "it'll": "it shall / it will",
    "it'll've": "it shall have / it will have",
    "it`ll`ve": "it shall have / it will have",
    "it's": "it has / it is",
    "it`s": "it has / it is",
    "let's": "let us",
    "let`s": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "ma`am": "madam",
    "mayn`t": "may not",
    "might`ve": "might have",
    "mightn`t": "might not",
    "mightn't've": "might not have",
    "mightn`t`ve": "might not have",
    "must've": "must have",
    "must`ve": "must have",
    "mustn't": "must not",
    "mustn`t": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "mustn`t`ve": "must not have",
    "needn`t": "need not",
    "needn`t`ve": "need not have",
    "o`clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "oughtn`t": "ought not",
    "oughtn`t`ve": "ought not have",
    "shan`t": "shall not",
    "shant": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she had / she would",
    "she'd've": "she would have",
    "she'll": "she shall / she will",
    "she'll've": "she shall have / she will have",
    "she's": "she has / she is",
    "should've": "should have",
    "shouldn't": "should not",
    "sha`n`t": "shall not",
    "shan`t`ve": "shall not have",
    "she`d": "she had / she would",
    "she`d`ve": "she would have",
    "she`ll": "she shall / she will",
    "she`ll`ve": "she shall have / she will have",
    "she`s": "she has / she is",
    "should`ve": "should have",
    "shouldn`t": "should not",
    "shouldnt": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as / so is",
    "that'd": "that would / that had",
    "that'd've": "that would have",
    "that's": "that has / that is",
    "there'd": "there had / there would",
    "there'd've": "there would have",
    "there's": "there has / there is",
    "they'd": "they had / they would",
    "they'd've": "they would have",
    "they'll": "they shall / they will",
    "they'll've": "they shall have / they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "shouldn`t`ve": "should not have",
    "so`ve": "so have",
    "so`s": "so as / so is",
    "that`d": "that would / that had",
    "that`d`ve": "that would have",
    "that`s": "that has / that is",
    "there`d": "there had / there would",
    "there`d`ve": "there would have",
    "there`s": "there has / there is",
    "they`d": "they had / they would",
    "they`d`ve": "they would have",
    "they`ll": "they shall / they will",
    "they`ll`ve": "they shall have / they will have",
    "they`re": "they are",
    "they`ve": "they have",
    "to`ve": "to have",
    "wasn`t": "was not",
    "wasnt": "was not",
    "we'd": "we had / we would",
    "we`d": "we had / we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "we`d`ve": "we would have",
    "we`ll": "we will",
    "we`ll`ve": "we will have",
    "we`re": "we are",
    "we`ve": "we have",
    "weren't": "were not",
    "weren`t": "were not",
    "werent": "were not",
    "what'll": "what shall / what will",
    "what`ll": "what shall / what will",
    "whatll": "what shall / what will",
    "what'll've": "what shall have / what will have",
    "what`ll`ve": "what shall have / what will have",
    "what're": "what are",
    "what`re": "what are",
    "what's": "what has / what is",
    "what've": "what have",
    "what`s": "what has / what is",
    "what`ve": "what have",
    "when's": "when has / when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where has / where is",
    "where've": "where have",
    "who'll": "who shall / who will",
    "when`s": "when has / when is",
    "when`ve": "when have",
    "where`d": "where did",
    "where`s": "where has / where is",
    "where`ve": "where have",
    "who`ll": "who shall / who will",
    "wholl": "who shall / who will",
    "who'll've": "who shall have / who will have",
    "who`ll`ve": "who shall have / who will have",
    "whollve": "who shall have / who will have",
    "who's": "who has / who is",
    "who`s": "who has / who is",
    "who've": "who have",
    "who`ve": "who have",
    "why's": "why has / why is",
    "why`s": "why has / why is",
    "whys": "why has / why is",
    "why've": "why have",
    "why`ve": "why have",
    "will`ve": "will have",
    "will've": "will have",
    "won't": "will not",
    "won`t": "will not",
    "wont": "will not",
    "won't've": "will not have",
    "won`t`ve": "will not have",
    "would`ve": "would have",
    "would've": "would have",
    "wouldve": "would have",
    "wouldn`t": "would not",
    "wouldn't": "would not",
    "wouldnt": "would not",
    "wouldn`t`ve": "would not have",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had / you would",
    "y`all": "you all",
    "y`all`d": "you all would",
    "y`all`d`ve": "you all would have",
    "y`all`re": "you all are",
    "y`all`ve": "you all have",
    "you`d": "you had / you would",
    "youd": "you had / you would",
    "you'd've": "you would have",
    "you`d`ve": "you would have",
    "you'll": "you shall / you will",
    "you`ll": "you shall / you will",
    "youll": "you shall / you will",
    "you'll've": "you shall have / you will have",
    "you`ll`ve": "you shall have / you will have",
    "you're": "you are",
    "you`re": "you are",
    "youre": "you are",
    "you've": "you have",
    "youve": "you have",
    "you`ve": "you have",
    }
    for word in text.split():
        #print("/"+str(word)+"/")
        if word.lower() in contractions:
            #print(word)
            text = text.replace(word, contractions[word.lower()])
            #print(text)
    return text 
def data_preprocessing(data):
    step=1
    print(data.head())
    print('Step {}/10: Renaming columns and removing "irrelevant" sentiments'.format(step))
    data = pd.DataFrame({"Tweet": data['text'],
                    "Sentiment": data['sentiment']})
    data.dropna(inplace=True)
    data=data.drop_duplicates(keep = 'first')
    step+=1
    print(data["Tweet"][49])
    print('Step {}/10: Converting categorical data into integer format'.format(step))
    data["Sentiment"] = np.array(list(map(lambda y: 2 if y=='positive'
                                            else (1 if y=='neutral' else 0),data["Sentiment"])))
    step+=1
    print('Step {}/10: Converting to lower case'.format(step))
    data['Tweet'] = [i.lower() for i in data['Tweet']]
    step+=1
    print(data["Tweet"][49])
    print('Step {}/10: Removing punctuation'.format(step))
    data['Tweet'] = [func_not(i) for i in data['Tweet']]
    data['Tweet'] = [i.translate(str.maketrans('', '', string.punctuation)) for i in data['Tweet']]
    step+=1
    print(data["Tweet"][49])
    print('Step {}/10: Removing single words'.format(step))
    data['Tweet'] =  [re.sub(r"\b[a-zA-Z]\b", "", i) for i in data['Tweet']]
    step+=1
    print(data["Tweet"][49])
    print('Step {}/10: Removing mentions and hastag symbols'.format(step))
    data['Tweet'] =  [re.sub(r"@[a-zA-Z0-9]+|\#[a-zA-Z0-9]","",str(i)) for i in data['Tweet']]
    step+=1
    print(data["Tweet"][49])
    print('Step {}/10: Removing URLs'.format(step))
    data['Tweet'] =  [re.sub(r'http\S+', "", str(i)) for i in data['Tweet']]
    data['Tweet'] =  [re.sub(r"[0-9]","",i) for i in data['Tweet']]
    step+=1
    print(data["Tweet"][49])
    print('Step {}/10: Converting emoji to words'.format(step))
    data['Tweet']= [remove_emoji(i) for i in data['Tweet']]
    step+=1
    print(data["Tweet"][49])
    print('Step {}/10: Lemmatization'.format(step))
    data['Tweet'] =  [lemmatize_sentence(i) for i in data['Tweet']]
    step+=1
    print(data["Tweet"][49])
    print('Step {}/10: Removing stop words'.format(step))
    data['Tweet']= [remove_stopwords(i) for i in data['Tweet']]
    print(data["Tweet"][49])
    # data['Tweet']= [" ".join(i) for i in data['Tweet']]
    print(data["Tweet"][49])
    print(data.head(20))
    return data


if __name__ == '__main__':
    print('Data preprocessing')
    train_data2=pd.read_csv('twitter_train2.csv', encoding='latin1')
    test_data2=pd.read_csv('twitter_test2.csv',encoding='latin1')
    #print(len(test_data1), len(test_data2))
    test_data=data_preprocessing(test_data2)
    train_data=data_preprocessing(train_data2)
    print('Data saving')
    train_data.to_csv('train2.csv', encoding='utf-8', index=False)
    test_data.to_csv('test2.csv', encoding='utf-8', index=False)
    #print(len(test_data), test_data1['Sentiment'].value_counts())
    print('Completed')
#     test_data=pd.read_csv('test_exp.csv')
#     train_data=pd.read_csv('../datasets/prepared_train.csv')
#     print(test_data)
#     ResultsEvaluation.Results( test_data['Tweet'], test_data['Sentiment'],train_data['Tweet'])
    

    
