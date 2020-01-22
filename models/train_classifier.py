import sys
import numpy as np
import pandas as pd
from nltk import pos_tag
import re
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet') # download for lemmatization
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

def load_data(database_filepath):
    """
    load the data set and return the variables
    for the training model
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM InsertTableName', engine)
    X=df["message"]
    Y=df.iloc[:,4:]
    return(X,Y,Y.columns)
    pass


def tokenize(text):
    """
    tokenize the message
    """
    text=text.lower()
    text=re.sub(r"[^a-zA-Z0-9]"," ",text)
    words =word_tokenize(text)
    words =[w for w in words if w not in stopwords.words("english")]
    word_lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    return word_lemmed
    pass


def build_model():
    pipeline =Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('moc', MultiOutputClassifier(estimator=RandomForestClassifier()))
    ])
    parameters = {
        'vect__max_features': (None, 5000),
        'moc__estimator__n_estimators': [20, 30],
        'moc__estimator__min_samples_split': [2, 3]}
    cv = GridSearchCV(pipeline, param_grid=parameters)
    return cv
    
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    
    y_pred = model.predict(X_test)
    for i in range(0, len(category_names)):
          print(category_names[i])
          print(classification_report(np.array(Y_test)[:, i], y_pred[:, i]))
    pass


def save_model(model, model_filepath):
      with open(model_filepath, 'wb') as file:  
           pickle.dump(model, file)
      pass


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