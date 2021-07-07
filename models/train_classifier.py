import sys
import os
import pandas as pd
import numpy as np
import nltk
import joblib

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score

nltk.download(['punkt', 'wordnet', 'stopwords'])


def load_data(database_filepath):
    '''
    Load data from the given file path

    :param database_filepath: The database filepath
    :returns X (messages), Y (Disaster categories), disaster category names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    table_name = database_filepath.split(os.path.sep)[-1].split('.')[0]
    df = pd.read_sql_table(table_name, engine)
    
    X = df.message
    Y = df.iloc[:, 4:]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    '''
    Tokenize the given text into a list of words
    It normalised each word and removes stop words
    
    :param text: The text string
    :returns a list of normalised words
    '''
    stop_words = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()

    text = text.lower()
    words = [w.strip() for w in word_tokenize(text)]
    normalised_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    
    return normalised_words

def build_model():
    '''
    It builds a pipeline of preprocessing and model/estimator
    
    :returns a sklearn estimator object 
    '''
    pipeline = Pipeline([
        ('vectorizer', TfidfVectorizer(tokenizer=tokenize, min_df=3)),
        ('estimator', MultiOutputClassifier(OneVsRestClassifier(SVC(kernel='linear'))))
    ])
    
    params = {
        'estimator__estimator__estimator__C': [0.1, 1, 10]
    }
    
    model = GridSearchCV(pipeline, params, 'f1_samples', cv=2)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    '''
    It evaluates the given model on the test data (X_test, Y_test)
    It will print the ratio of 1s, precision, recall and f1 scores
    
    :param model: The sklearn estimator object
    :param X_test: Test data input
    :parma Y_test: Test data label
    :param category_names: Test data label names
    '''
    Y_predict = model.predict(X_test)
    out = []
    for i, col in enumerate(Y_test):
        avg_opt = ('weighted' if i == 0 else 'binary')
        
        f = f1_score(Y_test[col], Y_predict[:, i], average=avg_opt)
        pre = precision_score(Y_test[col], Y_predict[:, i], average=avg_opt)
        rec = recall_score(Y_test[col], Y_predict[:, i], average=avg_opt)
        ratio = Y_test[col].sum() / len(Y_test[col])
        out.append([ratio, pre, rec, f])
    out_df = pd.DataFrame(np.array(out).T, columns=Y_test.columns, 
                        index=['true_ratio', 'precision', 'recall', 'f1_score'])
    print(out_df)


def save_model(model, model_filepath):
    '''
    It saves the given model in a disk
    
    :param model: Sklearn estimator object
    :param model_filepath: File name to store
    '''
    joblib.dump(model, model_filepath)
    


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