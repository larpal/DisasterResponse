import sys
import os
# import libraries
from sqlalchemy import create_engine
import pandas as pd
# nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# scikit-learn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
# pickle 
import pickle

def load_data(database_filepath):
    path = os.path.abspath(os.getcwd())
    print(path+database_filepath[7:])
    tmp_str = 'sqlite:///{}'.format(path + database_filepath[7:])
    engine = create_engine(tmp_str)
    table_name = engine.table_names()[0]
    df = pd.read_sql_table(table_name, engine)
    X = df.message
    Y = df.drop(columns=['id','message','original','genre'])
    category_names = Y.columns


    return X,Y, category_names


def tokenize(text):
    # tokenize text
    tokens = word_tokenize(text)
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok.lower().strip())
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():

    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf',MultiOutputClassifier(RandomForestClassifier(verbose = 1,n_jobs=-1,random_state=42)))
                         ])
    # specify parameters for grid search
    parameters = {'clf__estimator__max_features' : [None, 'sqrt'],
            	  'clf__estimator__n_estimators': [100, 500]
             		}
    # create grid search object
    pipeline_cv = GridSearchCV(pipeline, parameters)                        

    return pipeline_cv


def evaluate_model(model, X_test, Y_test, category_names):
    for idx, col in enumerate(category_names):
        print('For category {}:'.format(col))
        print(classification_report(y_test[col], y_pred[:,idx]))
    pass


def save_model(model, model_filepath):
    outfile = open(model_filepath,'wb')
    pickle.dump(model, outfile)
    outfile.close()


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