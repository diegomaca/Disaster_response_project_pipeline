import sys
# import libraries
#!pip install --upgrade scikit-learn
import logging
import re
import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix,f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

import sqlite3
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

#import warnings
#from sklearn.exceptions import UndefinedMetricWarning

#warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

#nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'stopwords'])
#my_stop_words = set(stopwords.words('english'))

def load_data(database_filepath):
    # load data from database
   engine = create_engine('sqlite:///'+ database_filepath)
   conn = sqlite3.connect(database_filepath)
   df = pd.read_sql_table('messages_relations', con = engine)
   X = df['message'] 
   Y = df[list(df.columns)[4:]]
   category_names = list(df.columns)[4:]
#   X = df.message.values 
#   Y = df[list(df.columns)[4:]].values
#   category_names = list(df.columns)[4:]
   pass
   return X,Y, category_names


def tokenized(text):
    text = text.apply(lambda x: x.lower()).apply(lambda x: re.sub(r'[^\w\s]', '', x))
    return text

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
     ('tfidf', TfidfVectorizer(tokenizer=tokenize, stop_words = 'english')),
     ('clf', MultiOutputClassifier(RandomForestClassifier()))
     ])

    parameters = {
    'tfidf__use_idf': (True, False),
    'clf__estimator__max_depth': [5, 7],
#    'clf__estimator__n_estimators': [100, 500]    
    }     
   
    model = GridSearchCV(pipeline, parameters, cv=3, n_jobs=-1, verbose=2)

    return model

def evaluate_model(model,x_test, y_test,  category):
    y_pred = model.predict(x_test)
    y_pred = pd.DataFrame(y_pred)
    
    clsReport = classification_report( y_test.values,y_pred.values, target_names = col_pred )
    return clsReport
    #print(clsReport)
#    print("Classification report:", clsReport)

def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
    pass


def main():
        # Configurar el logger
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]

        logging.info('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        
        logging.info('Building model...')
        model = build_model()
        
        logging.info('Training model...')
        model.fit(X_train, Y_train)
        
        logging.info('Evaluating model...')
        clsReport = evaluate_model(model, X_test, Y_test, category_names)
        logging.info('Metrics: {}'.format(clsReport))

        logging.info('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        logging.info('Trained model saved!')

    else:
        logging.info('Please provide the filepath of the disaster messages database as the first argument and the filepath of the pickle file to save the model to as the second argument.\n\nExample: python train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()