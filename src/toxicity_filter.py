import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # type: ignore
from sklearn.naive_bayes import MultinomialNB # type: ignore
from sklearn.metrics import classification_report # type: ignore
from typing import *

class ToxicityFilter:


    def __init__(self, text_train: pd.DataFrame, labels_train: pd.DataFrame):

        self.vectorizer = CountVectorizer(stop_words='english')#creates a vocabulary from the training text and removes stop words
        self.vectorizer.fit(text_train)
        self.vectorizer.transform(text_train)
        # self.vectorizer.fit_transform(text_train, labels_train) apparently this messes up and retrains everytime according to spec
        features = self.vectorizer.transform(text_train)

        self.classifier = MultinomialNB()#trains a Multinomial Naive Bayes Classifier
        self.classifier.fit(features, labels_train)
        #features are the training data
        #labels_train are the training labels
        return
        
    def classify (self, text_test: list[str]) -> list[int]:

        features = self.vectorizer.transform(text_test)#vectorize list of strings into a matrix of word counts
        predictions = self.classifier.predict(features)#predict the class of the text
        return list(predictions)

    def test_model (self, text_test: pd.DataFrame, labels_test: pd.DataFrame) -> tuple[str, dict]:

        prediction = self.classify(text_test.values.tolist())#predict the class of the text
        return (classification_report(labels_test,prediction, output_dict = False),#print the classification report
                classification_report(labels_test,prediction, output_dict = True))#print the classification report in a dictionary format
    
