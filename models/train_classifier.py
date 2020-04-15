import sys
import os
# import libraries
from sqlalchemy import create_engine
import pandas as pd
# nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('wordnet')
nltk.download('punkt')
# scikit-learn
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
# pickle 
import pickle

def load_data(database_filepath):
    """Loads data from an SQL databases and splits it into
    	messages, labels, and category names

        Args: 
            database_filepath (str): file path database

        Returns: messages, labels, category names 
	"""
	# get path of current folder
    path = os.path.abspath(os.getcwd())
    # get total path to database
    tmp_str = 'sqlite:///{}'.format(path +'/'+ database_filepath)
	# create engine and extract table name
    engine = create_engine(tmp_str)
    table_name = engine.table_names()[0]
    df = pd.read_sql_table(table_name, engine)
    # extract messages, labels, and category names
    X = df.message
    Y = df.drop(columns=['id','message','original','genre'])
    category_names = Y.columns

    return X,Y, category_names


def tokenize(text):
    """NLP with four steps:
    	- tokenizes
    	- change to lower case
    	- strip whitespaces
    	- lemmatize

        Args: 
            text (str)

        Returns: tokens (list)
	"""
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
    """Build a machine learning classifier for multiple binary classication problems.
    	The pipeline consists of
    		- scikit-learn CountVectorizer using above defined tokenize function
    		- scikit-learn TfidfTransformer
    		- MultiOutput Logistic Regression classifier
		The model uses GridsearchCV to tune the parameters ngram_range of the 
		CountVectorizer and use_idf of the TfidfTransformer.

        Args: None

        Returns: pipeline
	"""
    pipeline = Pipeline([('vect', CountVectorizer(tokenizer = tokenize)),
                         ('tfidf', TfidfTransformer()),
                         ('clf',MultiOutputClassifier(LogisticRegression(solver='sag',random_state=42,max_iter = 500)))
                         ])
    # specify parameters for grid search
    parameters = {'vect__ngram_range' : [(1,1), (1,2)],
                 'tfidf__use_idf': [True, False]
                 }
    # create grid search object
    pipeline_cv = GridSearchCV(pipeline, parameters,cv=3, verbose = 1, n_jobs=-1)                        

    return pipeline_cv


def evaluate_model(model, X_test, Y_test, category_names):
	"""Scores a model using scklearn.metrics.classification_report. Reports
		are printed for each category.
		
		Args: 
			model: sklearn model
			X_test (pd.DataFrame): test data
			Y_test (pd.DataFrame): test labels
			category_names (list): category names
			
		Returns: 
			None
	"""
    Y_pred = model.predict(X_test)
    for idx, col in enumerate(category_names):
        print('For category {}:'.format(col))
        print(classification_report(Y_test[col], Y_pred[:,idx]))
    pass


def save_model(model, model_filepath):
    """"Saves model using pickle.
    
        Args:
        	model: model to be saved
        	model_filepath: file path
        
        Returns: 
        	None
    
    """"
    outfile = open(model_filepath,'wb')
    pickle.dump(model, outfile)
    outfile.close()


def main():
    """ Runs the ML pipeline """
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=42)
        
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
