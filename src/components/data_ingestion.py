
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass # This is used to create classes with default values for attributes

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer




@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv") # This is the path where the training data will be saved
    test_data_path: str=os.path.join('artifacts',"test.csv") # This is the path where the test data will be saved
    raw_data_path: str=os.path.join('artifacts',"podcast_data.csv") # This is the path where the raw data will be saved 

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()   # This will create an instance of the DataIngestionConfig class with default values for the attributes

    def initiate_data_ingestion(self):


        logging.info("Entered the data ingestion method or component")

        try:
            df=pd.read_csv('notebook/data/podcast_dataset.csv') # This will read the dataset from the specified path
            # The path should be relative to the current working directory or absolute path

            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) # This will create the directory if it does not exist
            # exist_ok=True means that if the directory already exists, it will not raise an error

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("Train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # This will save the training data to the specified path
            # index=False means that the index column will not be saved in the CSV file

            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) # This will save the test data to the specified path
            # index=False means that the index column will not be saved in the CSV file

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path, # This will return the path where the training data is saved
                self.ingestion_config.test_data_path    # This will return the path where the test data is saved

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    
    obj=DataIngestion() # This will create an instance of the DataIngestion class
    train_data,test_data=obj.initiate_data_ingestion() # This will call the initiate_data_ingestion method of the DataIngestion class

    data_transformation=DataTransformation() 
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data) # This will call the initiate_data_transformation method of the DataTransformation class
    # This will return the transformed training and test data along with the path where the preprocessor object is saved
    print("Data transformation completed successfully!")
    print(f"Train Array: {train_arr.shape}")
    print(f"Test Array: {test_arr.shape}")
    print(f"train dataset path: {train_arr[0]}")
    
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))



