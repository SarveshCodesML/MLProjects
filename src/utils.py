import sys,os
import pandas as pd
import numpy as np
import dill

from src.exception import Custom_Exception

def save_obj(file_path,obj):
    try:
        dir_name = os.path.dirname(file_path)

        os.makedirs(dir_name,exist_ok=True)
        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)
    except Exception as ex:
        raise Custom_Exception(ex,sys)