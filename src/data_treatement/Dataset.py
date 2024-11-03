from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas_profiling import ProfileReport

from scipy import stats
from scipy.stats.mstats import winsorize

import numpy as np

from helpers.DatasetHelper import DatasetHelper
from helpers.ContentHelper import ContentHelper

from sklearn.impute import KNNImputer, SimpleImputer
class Dataset:
    
    def __init__(self, original_data: DataFrame, target: str, verbose= False):        
      
        DatasetHelper.normalize_columns_name(original_data)

        self.__original_data = original_data.copy(deep=True)
        self._data = original_data.copy(deep=True)      
        self._has_many_header = isinstance(original_data.columns, pd.MultiIndex)
        self.__target = self.__find_multiindex(target,1)
        self._sections = []

        self._set_sections()    

        if verbose:
            print("Details of your Database")    
            #region Data balacing        

            _classes = self._data[self.__target].value_counts(normalize=True) * 100

            print("Percent of class:")
            print(_classes)

            #endregion

            #region Outliers

            self.__cols_outliers = DatasetHelper.find_columns_with_outliers(self._data)
            if any( self.__cols_outliers):
                print("Columns with outliers:")
                print("\n".join(self.__cols_outliers))

            #endregion
                
    def generate_data_report(self,path_to_save_report=None):
        profile = ProfileReport(self._data, title="Profiling Report")
        report = profile.to_html()   
        
        if path_to_save_report is not None:
            f = open(f"{path_to_save_report}.html", "w")
            f.write(report)
            f.close()
        
        return profile.to_notebook_iframe()

    def _typesToX(self):
        return ['int64', 'float64','int32']
    
    def _set_sections(self):
        if self._has_many_header:
            self._sections = np.unique(self._data.columns.get_level_values(0).values)   
    
    def get_has_many_header(self)-> bool:
        return self._has_many_header
    
    def get_sections(self)-> list:
        return self._sections

    def remove_duplicates(self,use_original_data: False):
        if use_original_data:
            self._data = self.__original_data.drop_duplicates(ignore_index=True)
        else:
            self._data = self._data.drop_duplicates(ignore_index=True)

    def __count_na(self, axis=1):
        neg_axis = 1 - axis
        count   = self._data.isna().sum(axis=neg_axis)
        percent = self._data.isna().mean(axis=neg_axis) * 100
        df_na   = pd.DataFrame({'percent':percent, 'count':count})
        index_name = 'column' if axis==ContentHelper.const_axis_column() else 'index'
        df_na.index.name = index_name
        df_na.sort_values(by='count', ascending=False, inplace=True)
        return df_na

    def print_na(self, axis=1):
        df_na = self.__count_na(axis=axis)
        #
        max_rows = pd.get_option('display.max_rows', None)
        pd.set_option('display.max_rows', None)
        print(df_na)
        pd.set_option('display.max_rows', max_rows)

    def drop_na(self, axis=0, percent=30.0, show_dropped=True):
        df_na = self.__count_na(axis=axis)
        to_drop = df_na[df_na['percent'] > percent].index
        if show_dropped:
            print(to_drop)
        
        self._data = self._data.drop(labels=to_drop, axis=axis)
        self._set_sections()  

    def resolve_missing_data(self, startegy='mean' ,use_original_data= False):       
        imputer = SimpleImputer(fill_value=np.nan, startegy=startegy)

        if use_original_data:
            self._data = self.__original_data

        self._data = imputer.fit_transform(self._data)

    def remove_outliers(self, method_remove= "limit_method", use_original_data= False):
        if method_remove is None:
            raise AttributeError("method_remove is not null")
        
        if use_original_data:
            self._data = self.__original_data
       
        for col in self.__cols_outliers:

            if method_remove == "limit_method":

                min_limit,max_limit =DatasetHelper.find_outliers_limit_IQR(self._data[col])
                
                #add +1 for remove all outliers, because if have a a big difference between limits and values outliers, still go have new outliers 
                percentile_min = (stats.percentileofscore(self._data[col], min_limit)+1)/100
                percentile_max = (100-stats.percentileofscore(self._data[col], max_limit)+1)/100
                self._data[col] = winsorize(self._data[col], limits=[percentile_min, percentile_max])

            elif method_remove == 'log_transformation':

                self._data[col] = np.log(self._data[col])

            elif method_remove == 'mean_value':

                mean = np.mean(self._data[col])
                df_aux= DatasetHelper.find_outliers_IQR(self._data[col]).dropna(axis=0,how='all')
                self._data.loc[self._data[col].isin(df_aux),col] = mean
            else:
                raise AttributeError("method_remove not exists")
   

    def clean_data(self,cols_to_drop:list[str] = [],cols_date:list[str] = [], cols_levels=0, try_convert_values= False, use_original_data= False):
        if use_original_data:
            self._data = self.__original_data
        
        if cols_to_drop is not None and any(cols_to_drop):
            self._data.drop(cols_to_drop,axis=1,inplace=True,level=cols_levels)
            self._set_sections()

        if cols_date is not None and any(cols_date):             
            for i in range(len(cols_date)):
                if isinstance(cols_date[i], str):
                   cols_date[i]= self.__find_multiindex(cols_date[i], cols_levels)
                else:
                    cols_date[i][0]= self.__find_multiindex(cols_date[i][0], cols_levels)

            ContentHelper.convert_datetime(self._data,cols_date)

        if try_convert_values:
            ContentHelper.try_convert_object_values(self._data)  

    def convert_cols_values(self, cols:list[str] = [""] ):
        cols = [self.__find_multiindex(col,1) for col in cols]
        ContentHelper.convert_cols_values(self._data,cols)
    
    def get_X(self, section: str= None)->DataFrame:
        if section is None and not self._has_many_header:
            numeric_cols = [cname for cname in self._data.columns if self._data[cname].dtype in self._typesToX()]
            X = self._data[numeric_cols].copy()
        else:
            numeric_cols = [cname for cname in self._data[section].columns if self._data[section][cname].dtype in self._typesToX()]
            X = self._data[section][numeric_cols].copy()
        
        if self.__target not in numeric_cols:           
            return X 
        else:
            return X.drop([self.__target],axis=1,inplace=False)
    
    def get_Y(self)->DataFrame:   
        return self._get_Y(self.__target)
    
    def _get_Y(self, target)->DataFrame:
        if self._data[target].dtype not in self._typesToX():
            self.convert_cols_values([target])
            
        return self._data[target]
    
    def impute_cols_na(self, method="knn", n_neighbors=5):
        if method == "knn":
            imputer = KNNImputer(n_neighbors=n_neighbors)            
        else:
           imputer = SimpleImputer(fill_value=np.nan, startegy=method)
        
        self._data = imputer.fit_transform(self._data) 


    def __find_multiindex(self, col, col_level=0):
        if isinstance(self._data.columns, pd.MultiIndex):
            for idx in self._data.columns:       
                if col in idx[col_level]:
                    return idx
        else:
            for idx in self._data.columns:       
                if col in idx:
                    return idx
        return None