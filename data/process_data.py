import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Reads the data from messages and categories files

    Args:
    messages_filepath: csv. The file that contains the disaster messages
    categories_filepath: csv. The file that contains the categories each messages falls into 
    (36 categories in total - each messages is assigned 1 or 0 for each category)

    Returns:
    A dataframe which is a merge of the two input files.
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    return pd.merge(messages,categories,on='id')

def clean_data(df):
    
    """Cleans the merged dataframe that the previous function created

    Args:
    df: The dataframe to be cleaned

    Returns:
    The clean dataframe where a separate column for each category has been created and assigned 0-1 values 
    """
    
    # creates a column for each category (therefore a   dataframe) 
    categories = df.categories.str.split(pat=';',expand=True)  
    
    # the values for these columns are of the form category-x (x=0-1) (e.g related-0, request-1)
    # the first row of the new dataframe will be used to extract the column names for the categories
    row = categories.iloc[0]  
    
    # apply lambda function that takes everything up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda x: x[:-2])   
        
    categories.columns = category_colnames  # rename the columns of `categories`
    
    # extract the number at the end (0-1)
    for column in categories.columns:
        categories[column] = categories[column].apply(lambda x: x[len(x)-1:len(x)])   
    
    # convert column from string to numeric
    for column in categories.columns:
        categories[column] = pd.to_numeric(categories[column])  
    
    # drop the original categories column from `df`    
    df.drop('categories',axis=1,inplace=True)  
    
    # concatenate the original dataframe with the new `categories` dataframe
    df=pd.concat([df,categories],axis=1)  

    # drop duplicates
    df.drop_duplicates(inplace=True)  

    # some categories were found to have a value of 2. They should change to 1 - lambda function applied
    for column in categories.columns:
        df[column]=df[column].apply(lambda x: 1 if x > 1 else x)    
         
    return df 

def save_data(df, database_filename):
    """Saves the clean dataset as a table in a specific sqlite database (DisasterResponse.db)

    Args:
    df: The cleaned dataframe 
    database_filename: The desired name of the table within the database
   
    Returns:
    It doesn't need to return anything as it just stores the dataset in the database table.
    """
        
    engine = create_engine('sqlite:///'+database_filename)  # create a connection to the database DisasterResponse.db
    df.to_sql('Disaster_category_messages', engine, index=False,if_exists='replace')  # save the dataframe as a table

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