import os
import sys
from src.logger import logging
import numpy as np 
import pandas as pd
# import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):

    # This function saves an object to a file using pickle or dill.
    # It creates the directory if it does not exist and handles exceptions.
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models,param):

    # This function evaluates multiple machine learning models using GridSearchCV for hyperparameter tuning.
    # It trains each model on the training data, predicts on both training and test data,   
    # and calculates the R-squared score for both sets.
    # It returns a report containing the R-squared scores for each model on the test data
    
    try:
        report = {}

        for i in range(len(list(models))):

            logging.info(f"Training model: {list(models.keys())[i]}")

            model = list(models.values())[i]  # This will get the model from the dictionary using the index

            para=param[list(models.keys())[i]] # This will get the parameters for the model from the dictionary using the model name

            logging.info(f"Parameters for {list(models.keys())[i]}: {para}")
            # Using GridSearchCV for hyperparameter tuning
            # cv=3 means 3-fold cross-validation, n_jobs=1 means using one CPU core for training, verbose=1 means showing progress
            gs = GridSearchCV(model,para,cv=3, n_jobs=1, verbose=1)

            gs.fit(X_train,y_train)
            logging.info(f"Best parameters for {list(models.keys())[i]}: {gs.best_params_}")
            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)
            logging.info(f"Model {list(models.keys())[i]} trained successfully")
            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report
        #return report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)