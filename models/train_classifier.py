import sys
import nltk
nltk.download(['punkt', 'wordnet','stopwords'])
# import libraries
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.metrics import classification_report
import pickle




def load_data(database_filepath):
    '''
    INPUT:
    database_filepath - the filepath of the database that contains needed data
    
    OUTPUT:
    X - Column containing the messages that is being used as a feature in predicition model
    Y - Multiple columns containing labels of different categories
    category_names - list of categories
    
    Reads data from messages table of SQL Database and creates the feature and label DataFrames.
    '''
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    conn = sqlite3.connect(database_filepath)

    df = pd.read_sql('SELECT * FROM messages', con=conn)

    # create feature and label dataframes
    columns_features = ['id','message','original','genre']
    
    X = df['message']
    Y = df.drop(columns_features,axis=1)

    #drop the category child_alone due to the LinearSVC classifier that cannot handle only one class
    Y = Y.drop('child_alone', axis=1)
    
    # get category names
    category_names = Y.columns
    
    return X,Y,category_names
    


def tokenize(text):
    '''
    INPUT:
    text - message text of a particular row
    
    OUTPUT:
    clean_tokens - a list of tokens created from message text 
    
    Extracts information from message text by replacing URLs, normalizing text, removing stopwords 
    and punctuation, genrating word tokens and lemmatizing them
    '''

    # define regex for urls
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

    # remove all urls in text
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "")

    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    # normalization
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenization
    words = word_tokenize(text)
    
    # stop words removal
    stop = stopwords.words("english")
    words = [t for t in words if t not in stop]

    # Lemmatization
    lemm = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    for word in lemm:
        token = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(token)

    return clean_tokens
 


def build_model():
    '''
    OUTPUT:
    cv - a result of GridSearch containing the best parameters for the pipeline
    
    Creates a pipeline that contains a CountVectorizer, a TfidfTransformer and a MultiOutputClassifier
    using LinearSVC(), defines parameters for the GridSearch and performs GridSearch.
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LinearSVC()))
    ])
    
    parameters = {
        'clf__estimator__C': [1.0, 0.5],
        'clf__estimator__max_iter': [500,1000,2000]
    }
    cv = GridSearchCV(pipeline,param_grid=parameters)
   
    return cv

def print_score_values(y_test, y_pred):
    '''
    INPUT:
    y_test - a dataframe containing the labels of the test data
    y_pred - a dataframe containing the predicted labels
        
    Iterates through the columns and prints the classification report using the test labels 
    and predicted labels, prints the accuracy of the model.
    '''

    idx = 0
    for column in y_test:
        print(column)
        print(classification_report(y_test[column], y_pred[column]))
        idx += 1
    accuracy = (y_pred == y_test.values).mean()
    print('Accuracy of the model:')
    print(accuracy)


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    INPUT:
    model - trained model that has to be evaluated
    X_test - dataframe containing the test data
    Y_test - a dataframe containing the labels of the test data
    category_names - a list containing names of categories


    Predicts the labels using the test dataframe and creating a new Dataframe out 
    of the predicted labels, calls the function that prints the score values of model
    '''
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred)
    Y_pred_df.columns = Y_test.columns
    
    print_score_values(Y_test, Y_pred_df)


def save_model(model, model_filepath):
    '''
    INPUT:
    model - trained model that has to be saved
    model_filepath - filepath to save the model at
        
    Saves the model as a pickle file under the given filepath
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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

        best_model = model.best_estimator_
        
        print('Evaluating model...')
        evaluate_model(best_model, X_test, Y_test, category_names)

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