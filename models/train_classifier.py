import sys
import nltk
nltk.download(['punkt', 'wordnet'])
# import libraries
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sqlite3
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
import pickle




def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))

    conn = sqlite3.connect(database_filepath)

    df = pd.read_sql('SELECT * FROM messages', con=conn)

    columns_features = ['id','message','original','genre']
    
    X = df['message']
    Y = df.drop(columns_features,axis=1)
    
    category_names = Y.columns
    
    return X,Y,category_names
    


def tokenize(text):
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for word in words:
        token = lemmatizer.lemmatize(word).lower().strip()
        clean_tokens.append(token)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'clf__estimator__n_estimators': [10,50,100],
        'clf__estimator__max_depth': [2,4],
    }
    cv = GridSearchCV(pipeline,param_grid=parameters)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred)
    Y_pred_df.columns = Y_test.columns
    
    
    for col in Y_test.columns:
        accuracy = accuracy_score(Y_test[col],Y_pred_df[col])
        precision = precision_score(Y_test[col],Y_pred_df[col],average='macro')
        recall = recall_score(Y_test[col],Y_pred_df[col],average='macro')
        f1score = f1_score(Y_test[col],Y_pred_df[col],average='macro')
        print(col)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1-Score", f1score)
        


def save_model(model, model_filepath):
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