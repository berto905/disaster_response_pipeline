'''
Project Name: DISASTER RESPONSE PIPELINE
Stage: Machine Learning
Argument:
    A cleaned SQLite database responses file
Output:
    Trained classifier
'''

# Import libraries
import sys

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pickle

# Download NLP packages
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    '''Load data from database
    
    Args:
        database_filepath (str): path to SQLite database file
    
    Returns:
        X: features 
        y: labels
        categories: category names
    '''
    
    # Create a dataframe from SQLite database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM messages", engine)
    
    # Extract features, labels, and category names from the dataframe
    X = df['message'].values 
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    category_names = y.columns
    y = y.values

    return X, y, category_names
    
def tokenize(text):
    '''Normalize, tokenize and lemmatize a text string
    
    Args:
        text (str): text to tokenize
    
    Returns:
        tokens: tokenized text
    '''
    # Load the stopwords
    stop_words = stopwords.words('english')
    
    # Instantiate the lemmmatizer
    lemmatizer = WordNetLemmatizer()

    # Normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens
    
def build_model():
    '''Build the classifier model using grid search
    
    Args: none
    
    Returns:
        classifier
    '''
    # Instantiate the ML pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(), n_jobs=1))
    ])
    
    parameters = {
        'vect__ngram_range': ((1,1), (1,2)),
        'tfidf__use_idf': [True, False],
        'clf__estimator__min_samples_split': [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''Test the model and report F1 score, precision and recall
    
    Args:
        model: trained classifier
        X_test: test features
        y_test: test labels
        category_names: array of category names
    '''
    y_pred = model.predict(X_test)

    for i in range(len(category_names)):
        print('Label:', category_names[i])
        print(classification_report(Y_test[:, i], y_pred[:, i]))

def save_model(model, model_filepath):
    '''Export the model as a pickle file
    
    Args:
        model: trained model
        model_filepath: path to saved pickle file location
    '''
    
    pickle_out = open(model_filepath, 'wb')
    pickle.dump(model, pickle_out)
    pickle_out.close()
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()