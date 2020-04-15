import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    
    return df


def clean_data(df):

    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';',expand=True)
    # select the first row of the categories dataframe
    row = list(categories.iloc[0])
    category_colnames = list(categories.iloc[0].apply(lambda  x : x[:-2]))
    categories.columns = category_colnames
    
    # converting values to 0 and 1
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x : x[-1])
    # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop old categories column and add new set of columns
    df.drop(columns='categories',inplace = True)
    df = pd.concat([df,categories], axis=1)
    
    # drop duplicates
    df = df.drop_duplicates()
    
    # drop columns with only one distinct value
    for col in df:
        if df[col].nunique() == 1:
            df.drop(columns=col, inplace=True)
            
    return df


def save_data(df, database_filename):
    tmp_str = 'sqlite:///{}'.format(database_filename)
    engine = create_engine(tmp_str)
    df.to_sql(database_filename, engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
