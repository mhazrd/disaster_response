import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    It loads data of messages and categories.
    It returns a single dataframe where the two data are joined

    :param messages_filepath: Filepath of messages csv
    :param categories_filepath: Filepath of categories csv
    :returns: a dataframe of messages and categories 
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.merge(messages, categories, on='id')
    return df 


def clean_data(df):
    '''
    It preprocesses the given dataframe as:
      1. Decompose the categories column values into seperate columns
      2. Drop duplicate rows

    :param df: Dataframe of messages and categories 
    :returns: a preprocessed dataframe
    '''
    # Create columns for each kind of category
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]

    category_colnames = row.apply(lambda s: s[:-2])
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df = pd.concat((df.drop('categories', 1), categories), axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    # Binarize
    df.loc[df.related == 2, 'related'] = 1

    return df


def save_data(df, database_filename):
    '''
    It saves the given dataframe as sqlite database in disk

    :param df: Preprocessed dataframe of messages and categories 
    :param database_filename: Sqlite database filename
    '''
    engine = create_engine('sqlite:///' + database_filename)
    table_name = database_filename.split('.')[0]
    df.to_sql(table_name, engine, index=False, if_exists='replace')


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