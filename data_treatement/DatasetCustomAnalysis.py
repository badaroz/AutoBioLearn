from pandas import DataFrame
from data_treatement.DatasetByFile import DatasetByFile


class DatasetCustomAnalysis(DatasetByFile):
    def __init__(self, file_path: str, groups:dict,delimiter: None, verbose=False):
        super().__init__(file_path,"",delimiter,verbose)
        self.sections = groups.keys()
        self.has_many_header = len(groups.keys()) > 0

    def get_X(self, section:str= None)->DataFrame:
        cols_names =self.groups[section]["x_cols_names"]
       
        numeric_cols = [cname for cname in cols_names if self._data[cname].dtype in self._typesToX()]
        X = self._data[numeric_cols].copy()
        target = self.groups[section]["y_col_name"]

        if target not in numeric_cols:           
            return X 
        else:
            return X.drop([target],axis=1,inplace=False)
    
    def get_Y(self):
        pass

    def get_Y(self, section:str = None)-> DataFrame:
        self._get_Y(self.groups[section]["y_col_name"])