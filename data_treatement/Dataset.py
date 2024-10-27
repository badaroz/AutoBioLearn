from matplotlib import pyplot as plt
import pandas as pd
from pandas import DataFrame
from pandas_profiling import ProfileReport

from scipy import stats
from scipy.stats.mstats import winsorize

import numpy as np

from helpers.DatasetHelper import DatasetHelper
from helpers.ContentHelper import ContentHelper

from IPython.display import display, HTML


from sklearn.impute import KNNImputer, SimpleImputer
class Dataset:

    def __typesToX(self):
        return ['int64', 'float64','int32']
    
    def __init__(self, original_data: DataFrame, target: str, verbose= False):        
      
        DatasetHelper.normalize_columns_name(original_data)

        self.__Original_Data = original_data.copy(deep=True)
        self.__Data = original_data.copy(deep=True)
        self.__Target = target

        if verbose:
            print("Details of your Database")    
            #region Data balacing        

            _classes = self.__Data[self.__Target].value_counts(normalize=True) * 100

            print("Percent of class:")
            print(_classes)

            #endregion

            #region Outliers

            self.__cols_outliers = DatasetHelper.find_columns_with_outliers(self.__Data)
            if any( self.__cols_outliers):
                print("Columns with outliers:")
                print("\n".join(self.__cols_outliers))

            #endregion
                

    def data_analysis(self,path_to_save_report=None):
        profile = ProfileReport(self.__Data, title="Profiling Report")
        report = profile.to_html()
        
        display(HTML(report))
        
        if path_to_save_report is not None:
            f = open(f"{path_to_save_report}.html", "w")
            f.write(report)
            f.close()

       

    def remove_duplicates(self,use_original_data: False):
        if use_original_data:
            self.__Data = self.__Original_Data.drop_duplicates(ignore_index=True)
        else:
            self.__Data = self.__Data.drop_duplicates(ignore_index=True)

    def __count_na(self, axis=1):
        neg_axis = 1 - axis
        count   = self.__Data.isna().sum(axis=neg_axis)
        percent = self.__Data.isna().mean(axis=neg_axis) * 100
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
        
        self.__Data = self.__Data.drop(labels=to_drop, axis=axis)  

    def resolve_missing_data(self, startegy='mean' ,use_original_data= False):       
        imputer = SimpleImputer(fill_value=np.nan, startegy=startegy)

        if use_original_data:
            self.__Data = self.__Original_Data

        self.__Data = imputer.fit_transform(self.__Data)

    def remove_outliers(self, method_remove= "limit_method", use_original_data= False):
        if method_remove is None:
            raise AttributeError("method_remove is not null")
        
        if use_original_data:
            self.__Data = self.__Original_Data
       
        for col in self.__cols_outliers:

            if method_remove == "limit_method":

                min_limit,max_limit =DatasetHelper.find_outliers_limit_IQR(self.__Data[col])
                
                #add +1 for remove all outliers, because if have a a big difference between limits and values outliers, still go have new outliers 
                percentile_min = (stats.percentileofscore(self.__Data[col], min_limit)+1)/100
                percentile_max = (100-stats.percentileofscore(self.__Data[col], max_limit)+1)/100
                self.__Data[col] = winsorize(self.__Data[col], limits=[percentile_min, percentile_max])

            elif method_remove == 'log_transformation':

                self.__Data[col] = np.log(self.__Data[col])

            elif method_remove == 'mean_value':

                mean = np.mean(self.__Data[col])
                df_aux= DatasetHelper.find_outliers_IQR(self.__Data[col]).dropna(axis=0,how='all')
                self.__Data.loc[self.__Data[col].isin(df_aux),col] = mean
            else:
                raise AttributeError("method_remove not exists")
   

    def clean_data(self,cols_to_drop:list[str] = [],cols_date:list[str] = [], try_convert_values= False, use_original_data= False):
        if use_original_data:
            self.__Data = self.__Original_Data
        
        if cols_to_drop is not None and any(cols_to_drop):
            self.__Data.drop(cols_to_drop,axis=1,inplace=True)

        if cols_date is not None and any(cols_date):
            ContentHelper.convert_datetime(self.__Data,cols_date)

        if try_convert_values:
            ContentHelper.try_convert_object_values(self.__Data)  

    def convert_cols_values(self, cols:list[str] = [""] ):
        ContentHelper.convert_cols_values(self.__Data,cols)
    
    def get_X(self)->DataFrame:
        numeric_cols = [cname for cname in self.__Data.columns if self.__Data[cname].dtype in self.__typesToX()]
        X = self.__Data[numeric_cols].copy()
        
        if self.__Data[self.__Target].dtype not in self.__typesToX():           
            return X 
        else:
            return X.drop([self.__Target],axis=1,inplace=False)
    
    def get_Y(self)->DataFrame:
        if self.__Data[self.__Target].dtype not in self.__typesToX():
            self.convert_cols_values([self.__Target])
            
        return self.__Data[self.__Target]
    
    def input_na(self, method="knn", n_neighbors=5):
        if method == "knn":
            imputer = KNNImputer(n_neighbors=n_neighbors)
            self.__Data = imputer.fit_transform(self.__Data)
        else:
            self.__input_na_aux(method=method)


    def __input_na_aux(self,method):
        df_na = self.__count_na(axis=ContentHelper.const_axis_column())
        to_input = df_na[df_na['percent'] > 0]["column"].values
        for column in to_input:
            self.__Data[column].fillna(self.__get_value_to_input(method=method,column=column), inplace=True)
    
    def __get_value_to_input(self, method, column):
    
        if method == "mode":
            return self.__Data[column].mode()[0]
        elif method == "median":
             return self.__Data[column].median()
        elif method == "median":
           return self.__Data[column].median()