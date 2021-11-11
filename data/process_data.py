import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    INPUT:
    messages_filepath - filepath of the csv file containing the messages 
    categories_filepath - filepath of the csv file containing the categories 
    
    OUTPUT:
    df - a dataframe that merges the message data and the categories data on the id column
    
    Provides merged DataFrame containing the messages and the matching categories for further use in 
    prediction model
    '''

    #read from csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id')
    
    return df


def clean_data(df):
    '''
    INPUT:
    df - a dataframe with one column containig all category values
    
    OUTPUT:
    df - a clean dataframe consisting of one column per category and a binary value wheter the message
         of this particular row matches the category 
    
    Provides a clean DataFrame with one column per category and no duplicates.
    '''

    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(pat=';', expand=True))
    
    # select the first row of the categories dataframe
    row = categories.loc[0,:]
    
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2])
    
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str.slice(-1)

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df = df.drop(['categories'], axis=1)
        
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)

    # drop duplicates
    df = df.drop_duplicates()
        
    return df


def save_data(df, database_filename):
    '''
    INPUT:
    df - a dataframe that is saved in a database
    database_filename - the name of the file that contains database
    
    Creates a new engine and saves the dataframe to a SQL Database in the messages table
    '''
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df.to_sql('messages', engine, if_exists='replace', index=False)  


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