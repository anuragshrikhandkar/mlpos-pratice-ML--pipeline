import os 
import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
# from nltk.stem.porter PorterStemmer
#from sklearn.preprocessing import PorterStemmer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import string 
import nltk
nltk.download('stopswords')
nltk.download('punkt')


log_dir='logs'
os.makedirs(log_dir,exist_ok=True)


logger=logging.getLogger('data_processing')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_processing.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transform_text(text):
    """Transform the input text to converting it to lowercase tokenize removing stopwords and puncutation"""
    ps=PorterStemmer()
    text=text.lower()
    text=nltk.word_tokenize(text)
    text=[word for word in text if word.isalnum()]
    text=[word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    text=[ps.stem(word) for word in text]

    return " ".join(text)

def preprocess_df(df,text_columns='text',target_columns='target'):
    """preprocesses the DataFrame by encoding the target columns removing duplicts and transforming text columns """

    try:
        logger.debug('Strating preprocessing fro DataFrame')
        encoder=LabelEncoder()
        df[target_columns]=encoder.fit_transform(df[target_columns])
        logger.info('Target columns encoded')

        df=df.drop_duplicates(keep="first")
        logger.debug('Duplicate removed')
  
        df.loc[:, text_columns] = df[text_columns].apply(transform_text)
        logger.debug('Text column transformed')
        return df
    except KeyError as e:
        logger.error('columns not found :%s',e)
        raise
    except KeyError as e:
        logger.error('Error during text normalization:%s',e)
        raise

def main(text_columns='text',target_colums='target'):
    """main function to load raw data,preprocesses it and save the processed data""" 

    try:
        train_data=pd.read_csv('data/raw/train_data.csv')
        # ('./data/raw/train.csv')
        test_data=pd.read_csv('data/raw/test_data.csv')
        logger.info("Data Loaded successfully")

        ## transform data
        train_processed_data=preprocess_df(train_data,text_columns,target_colums)
        test_processed_data=preprocess_df(test_data,text_columns,target_colums)
        data_path=os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)

        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)

        logging.info("Processed data saved successfuly :%s",data_path)


    except FileNotFoundError as e:
        logger.error("'file not found error: %s",e)
    except pd.errors.EmptyDataError as e:
        logger.error('no data :%s',e) 
    except Exception as e:
        logger.error("failed to complete the data transformation process: %s",e)
        print(f"error:{e}")       
        

if __name__=="__main__":
    main()
    
    
        



