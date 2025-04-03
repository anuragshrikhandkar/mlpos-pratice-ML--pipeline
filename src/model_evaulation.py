import pandas as pd
import numpy as np
import os
import pickle 
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,roc_auc_score
import logging
import yaml
# from dvclive import dvclive

log_dir='logs'
os.makedirs(log_dir,exist_ok=True)
logger=logging.getLogger('model_evaulation')
logger.setLevel('DEBUG')

console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path=os.path.join(log_dir,'model_evaulation.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)



def load_models(file_path:str):
    try:
      with open(file_path,'rb') as file:
         models=pickle.load(file)
      logger.debug("models loaded from %s",file_path)
      return models
    except FileNotFoundError:
      logger.error('file not found: %s',file_path)
      raise
    except Exception as e:
       logger.error('unexpected error while loading the model:%s',e)
       raise

def load_data(file_path:str)->pd.DataFrame:
   """load data  form csv file"""
   try:
      df=pd.read_csv(file_path)
      logger.debug("Data loded for :%s",file_path)
      return df
   except FileNotFoundError:
      logger.error("file not found :%s",file_path)
      raise 
   except Exception as e:
      logger.error("unexpected error found :%s",e)
      raise 
   
def evaulate_model(clf,X_test:np.ndarray,y_test:np.ndarray)->dict:
   """Evaulation model turn into evaulation metrics """
   try:
      y_pred=clf.predict(X_test)
      y_pred_prob=clf.predict_proba(X_test)[:,1]

      accuracy=accuracy_score(y_test,y_pred)
      precision=precision_score(y_test,y_pred)
      recall=recall_score(y_test,y_pred)
      auc=roc_auc_score(y_test,y_pred_prob)

      metrics_dict={
         'accuracy':accuracy,
         'precision':precision,
         'recall':recall,
         'auc':auc
         }       
      logger.debug('Model Evaulation metrics completed')
      return metrics_dict
   except Exception as e:
      logger.error("error during model evaluation :%s",e)
      raise
# def evaluate_model():
#     print("Evaluating model...")
#     metrics = {"accuracy": 0.85}  # Example metric
#     with open("reports/metrics.json", "w") as f:
#         json.dump(metrics, f)
#     print("Evaluation complete.")

# if __name__ == "__main__":
#     evaluate_model()


def save_metrics(metrics:dict,file_path:str)->None:
   """Save The evaulation metrics to JSON file"""
   try:
       os.makedirs(os.path.dirname(file_path),exist_ok=True)
       with open(file_path,'w') as file:
         json.dump(metrics,file,indent=4)
       logger.debug('metrics saved t %s',file_path)
  
   except Exception as e:
      logger.error('error occured  while saving the metrics:%s',e)
      raise  
      


def main():
   try:
      clf=load_models('./models/model.pkl')
      test_data=load_data('./data/processed/test_tfidf.csv')

      X_test=test_data.iloc[:,:-1].values
      y_test=test_data.iloc[:,-1].values

      metrics=evaulate_model(clf,X_test,y_test)

      save_metrics(metrics,'reports/metrics.json')

   except Exception as e:
      logger.error("failed to completd model evaulation processed:%s",e)

      print(f"Error: {e}")
    
if __name__=="__main__" :
  main()   

    


