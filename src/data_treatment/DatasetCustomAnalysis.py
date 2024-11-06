from typing import overload
from pandas import DataFrame
from data_treatment.DatasetByFile import DatasetByFile


class DatasetCustomAnalysis(DatasetByFile):
    def __init__(self, file_path: str, groups:dict,delimiter: None, verbose=False):
        super().__init__(file_path,"",delimiter,verbose)
        self._sections_name = groups.keys()
        self._has_many_header = len(groups.keys()) > 0


    @overload
    def get_X(self, section:str= None)->DataFrame:
        cols_names =self.groups[section]["x_cols_names"]
       
        numeric_cols = [cname for cname in cols_names if self._data[cname].dtype in self._typesToX()]
        X = self._data[numeric_cols].copy()
        target = self.groups[section]["y_col_name"]

        if target not in numeric_cols:           
            return X 
        else:
            return X.drop([target],axis=1,inplace=False)
        
    @overload
    def _set_sections(self):
        pass

    @overload
    def get_Y(self):
        pass

    def get_Y(self, section:str = None)-> DataFrame:
        self._get_Y(self._data[self.groups[section]["x_col_name"]],self.groups[section]["y_col_name"])