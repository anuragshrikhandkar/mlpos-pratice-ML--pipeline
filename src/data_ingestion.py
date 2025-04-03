import os
from matplotlib import _preprocess_data
import pandas as pd
from sklearn.model_selection import train_test_split
import logging 
import yaml

log_dir="logs"
os.makedirs(log_dir,exist_ok=True)

## logging config 
logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


# def load_params(params_path:str)->dict:
#     """load params from yaml file"""

#     try:
#         with open(params_path,'r') as file:
#             params=yaml.safe_load(file)
#         logger.debug('File not found:%s',params_path)
#         return params 
        
        
#     except Exception as e:
#         logger.error("parameters retrived from %",params_path)
#         raise
#     except yaml.YAMLError as e :
#         raise
#     except Exception as e:
#         logger.error("unexpectd error:%s",e)
#         raise



def load_data(data_url:str)->pd.DataFrame:
    """load data from csv file"""
    try:
        df=pd.read_csv(data_url)
        logger.debug('data loaded form :%s',data_url)
        return df
    except pd.errors.ParseError as e:
        logger.error("failed to parse the csv file :%s",e)
        raise 
    except Exception as e:
        logger.error("unexpected error occured while loading data:%s",e)
        raise 

def preprocessing_data(df:pd.DataFrame)->pd.DataFrame:
    """prepere data processing"""

    try:
        df.drop(columns=["Unnamed: 2","Unnamed: 3","Unnamed: 4"],inplace=True)
        df.rename(columns={"v1":"target","v2":'text'},inplace=True)
        logger.debug("Data preprocessing completed")
        return df


    except Exception as e:
        logger.info("missing columns in data frame :%s",e)
        raise 

    except Exception as e:
        logger.error("unexpected error while preprocessing data:%s",e)
        raise

    #data save
def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    """save the train and test data"""
    try:
       raw_data_path=os.path.join(data_path,'raw')
       os.makedirs(raw_data_path,exist_ok=True)
       train_data.to_csv(os.path.join(raw_data_path,'train_data.csv'),index=False)
       test_data.to_csv(os.path.join(raw_data_path,'test_data.csv'),index=False)
       logger.debug("train,test data saved successfully %s",raw_data_path)



    except Exception as e:
         logger.error("unexpected error occured while saving the data") 
         raise 
    

    ##main

def main():
    try:
        #params=load_params(params_path='params_yaml') 
        #test_size=params['data_ingestion']['test_size']
        test_size=0.2 
        data_path='https://raw.githubusercontent.com/anuragshrikhandkar/DATASETS-MLPOS-PRATICE-/refs/heads/main/spam.csv'
        df=load_data(data_url=data_path)
        final_df = preprocessing_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == "__main__":
    main()




