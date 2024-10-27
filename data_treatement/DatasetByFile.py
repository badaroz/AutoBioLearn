import os
import pandas as pd
from pandas import DataFrame
from data_treatement.Dataset import Dataset

class DatasetByFile(Dataset):
    def __init__(self, file_path, target: str, delimiter: None, verbose=False):
        df: DataFrame

        file_extension = os.path.splitext(file_path)[1]

        if file_extension in [".xls",".xlsx"]:
            df = pd.read_excel(file_path)
        elif file_extension == ".csv":
              df = pd.read_csv(file_path,delimiter= delimiter)
        elif file_extension == ".txt":
             df = pd.read_csv(file_path, sep=delimiter)
        else:
            raise TypeError("Not support to this extesion")

        super().__init__(df, target,verbose)