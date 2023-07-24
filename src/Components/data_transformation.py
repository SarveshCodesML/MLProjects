import os,sys
from src.logger import logging
from src.exception import Custom_Exception
import pandas as pd
from dataclasses import dataclass
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.utils import save_obj

@dataclass
class DataTransofrmationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransofrmationConfig()

    def get_data_transformer_obj(self):
        try:
            numerical_columns = ['reading_score', 'writing_score']
            categorical_columns= ['gender','race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )

            logging.info("Numerical column scaling completed")

            logging.info("Categorical columns encoding completed")

            pre_processor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return pre_processor
        except Exception as ex:
            raise Custom_Exception(ex,sys)
    def initiate_datatransformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("The train and test data is imported in dataframe")

            logging.info("obtaining preprocessing object")

            preprocessing_object=self.get_data_transformer_obj()

            target_column_name = "math_score"

            numerical_columns = ['reading_score', 'writing_score']

            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying transformer for preprocessing")

            input_feature_train_arr = preprocessing_object.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_object.fit_transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_object
            )

            return (
                train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as ex:
            raise Custom_Exception(ex,sys)
