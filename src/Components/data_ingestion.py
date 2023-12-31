import os,sys
from src.exception import Custom_Exception
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.Components.data_transformation import DataTransformation
from src.Components.data_transformation import DataTransofrmationConfig
from src.Components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts','train.csv')
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','raw.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion component")
        try:
            df = pd.read_csv(r"Data\stud.csv")
            logging.info("Exported csv data to dataframe")
            
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train test split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of data is completed")

            return(self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)


        except Exception as ex:
            raise Custom_Exception(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    transform_obj = DataTransformation()
    train_array,test_array,_ = transform_obj.initiate_datatransformation(train_data,test_data)

    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_array,test_array))

