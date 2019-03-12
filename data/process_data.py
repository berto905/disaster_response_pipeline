'''
Project Name: DISASTER RESPONSE PIPELINE
Stage: Data Preparation
Argument:
    1) messages: disaster_messages.csv
    2) categories: disaster_categories.csv
Output:
    1) SQLite database (DisasterResponse.db)
    
'''

# Import libraries
import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories,
    convert and merge them into a pandas dataframe
    
    Arguments:
        messages_filepath: path to messages file
        categories_filepath: path to categories file
    Output:
        df: pandas dataframe
    '''
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    '''
    Clean pandas dataframe
    
    Arguments:
        df: dirty data
    Output:
        df: cleaned data
    '''
    # Split categories into separate category columns and rename the columns
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    first_row = categories.iloc[0]
    category_colnames = [name[:-2] for name in first_row]
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
    
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # Replace 'categories' column with new category columns
    df.drop('categories', axis=1, inplace=True) # drop 'categories'
    df = pd.concat([df, categories], axis=1) # concatenate original df with new category columns
    
    # Drop duplicates
    df.drop_duplicates(subset='id', inplace=True)
    
    return df
    
def save_data(df, database_filename):
    '''
    Save data into an SQLite database
    
    Arguments:
        df: clean pandas dataframe
    Output:
        database_filename:SQL database destination path
    '''
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, index=False)

def main():
    '''
    Main data processing function
    '''

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