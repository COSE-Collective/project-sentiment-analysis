from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
class RandomForest:
    def __init__(self, data):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(data["Tweet"], data["Sentiment"], test_size=0.3, random_state = 42)

    def Train(self):
        print('TF-IDF vectorization')
        tfidf = TfidfVectorizer(decode_error = 'replace', encoding = 'utf-8')
        tfidf.fit(self.X_train.values.astype('U'))
        self.X_train = tfidf.transform(self.X_train )
        self.X_test = tfidf.transform(self.X_test)
        RFClassifier = RandomForestClassifier()
        print('Random Forest training')
        RFClassifier.fit(self.X_train, self.Y_train)
        print('Model built successfully')
        print('Model saving')
        filename = 'src/Models/RandomForestClassifier/RandomForestClassifier.pkl'
        pickle.dump(RFClassifier, open(filename, 'wb'))
        print('Model saved')
        return self.X_test, self.Y_test