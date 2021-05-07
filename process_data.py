import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    
    """ Function to load datasetthat contain messages and categories and 
    merge them to get a dataframe.
    
    Input : file_paths for both message data and categories data
    Output : merged dataframe 'df' """
    
    #read messages and categories data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    #merge both the dataframes
    df = pd.concat([messages, categories], axis=1, join="inner")
    
    #drop duplicates
    df = df.T.drop_duplicates().T
    
    return df


def clean_data(df):
    """ Function to clean the dataset for classification,
    Split categories column in df dataframe delimiter ';'
    Create a dataframe 'categories' of 36 categories from dataframe with first row values as column names
    Keep only the last characters of the row values [0 or 1] and convert datatype to  numeric
    Replace 2 with 1 in column 'related` and drop the column `child_alone` as this column only have a single value
    Drop original categories column
    Merge categories, df dataframes and drop duplicate values
    
    Input: dataframe `df`
    Ouput: clean dataframe """
    # create a dataframe of the 36 individual category columns
    categories = pd.DataFrame(df['categories'].str.split(';',expand = True))

    # select the first row of the categories dataframe
    row = list(categories.iloc[0])

    # use this row to extract a list of new column names for categories.
    category_colnames =[(lambda x : x[:-2])(x) for x in row]

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Iterate through the category columns in df to keep only the last character of each string (the 1 or 0). 
    # Convert the string to a numeric value.
    for column in categories:
        categories[column] =   [x.strip()[-1] for x in categories[column]]
        categories[column] = pd.to_numeric(categories[column])

    # Replace 2 with 1 in column `related`
    # Drop column `child_alone` as all the values are the same 
    categories.drop(['child_alone'], axis=1, inplace=True)
    categories['related']=categories['related'].replace(2,1)

    # drop the original categories column from `df`
    df = df.drop(['categories'],axis=1)

    # merge `categories` and `df`
    df = pd.concat([df,categories],axis =1)
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """" Function to save the dataframe to sql_database with file name `Disaster_data`
    Input : dataframe, file_name
    Output : None """
    
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('disaster_data', engine, index=False,if_exists ='replace')
   

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