import sys
import nltk
nltk.download(['punkt','stopwords','wordnet'])
import pandas as pd
import numpy as np
import sqlalchemy as db
from sqlalchemy import create_engine

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV

import time
from sklearn.metrics import accuracy_score
import pickle

def load_data(database_filepath):
    """ Function to load data from sql database using the filepath and 
    create X and Y variables for further analysis
    
    input : database_filepath 
    output : arrays X and Y and list of naems of the categories """ 
    
    # load data from database
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('disaster_data',con=engine)

    # set column `message`as X and all other category columns as Y
    X = df['message']
    Y = df[df.columns[4:]]
    
    # get names of all the categories in a list
    category_names = list(Y.columns)
    
    return X,Y,category_names


def tokenize(text):
    """ Function to get the text into an analysable format,
    convert all the characters except alphabets and digits to spaces 
    convert all characters to lower lower case
    convert text to tokens 
    drop all the stopwords and lemmatize 
    
    input : message in the dataframe as string
     output : list of strings in message after processing """
    
    
    # Normalize by replacing all values other than alphabets and numbers with space and convert all the words to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()) 
    
    #tokenize the text
    words = word_tokenize(text)
    
    #drop all teh stopwords and lemmatize
    words = [w for w in words if w not in stopwords.words('english')]
    lemmed_words = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed_words
   


def build_model():
    """ Function to classify the each text message into different categories using Pipeline and MultiOutputClassifier
    feature : TFIDF 
    classifier : Linear SVC
    input : None
    output : model after getting best parameters using GridSearch CV """
   # specify model using pipeline 
    pipeline_SVC_CV = Pipeline([
    ('vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(LinearSVC(random_state = 42)))
    
])
    # specify parameters
    params = { "clf__estimator__C":[1,5,10]}
    
    # run GridSearchCV to get best parameters
    grid = GridSearchCV(pipeline_SVC_CV, param_grid=params,verbose=3,n_jobs=-1,cv=2) 
    
    return grid
   

def evaluate_model(model, X_test, Y_test, category_names):
    """ Function to evaluate the classification model using accuracy score and classification report to get 
    precision, recall and f1-score 
    
    input : classification model,
            array X_test, 
            array Y_test, 
            list category_names
            
    output : accuracy score and classification report """
    
    y_pred = model.predict(X_test)
    print('Accuracy: {:.2f}'.format(accuracy_score(Y_test, y_pred)))
    print(classification_report(Y_test,y_pred, 
                            target_names= category_names))
    


def save_model(model, model_filepath):
    """ Function to save the classification model as a pickle file
    input : classification model 
            str(file path of the model)
            
     output : None
   """      
    pkl_filename = "pickle_model.pkl"
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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