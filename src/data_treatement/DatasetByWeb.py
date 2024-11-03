import requests
from data_treatement.Dataset import Dataset
import pandas as pd


class DatasetByWeb(Dataset):
     def __init__(self, url:str, target: str, verbose= False):

        response = requests.get(url)
        data_json = response.json()
        df = pd.read_json(data_json)     

        super().__init__(df, target,verbose)
