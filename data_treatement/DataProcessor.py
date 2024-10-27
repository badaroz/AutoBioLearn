from data_treatement import Dataset
from decorators.DatasetDecorators import requires_dataset
from helpers.ContentHelper import ContentHelper


class DataProcessor:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @requires_dataset
    def convert_categorical_to_numerical(self, cols):
        self.dataset.convert_cols_values(cols)

    @requires_dataset
    def drop_cols_na(self, percent=30.0):
        self.dataset.drop_na(axis=ContentHelper.const_axis_column(), percent=percent, show_dropped=True)

    @requires_dataset
    def drop_rows_na(self, percent=10.0):        
        self.dataset.drop_na(axis=ContentHelper.const_axis_index(), percent=percent, show_dropped=True)
    
    @requires_dataset
    def print_cols_na(self):        
        self.dataset.print_na(axis=ContentHelper.const_axis_column())
    
    @requires_dataset
    def print_rows_na(self):        
        self.dataset.print_na(axis=ContentHelper.const_axis_column())

    @requires_dataset
    def remove_cols(self, cols:list[str] = []):      
        self.Dataset.clean_data(cols_to_drop =cols)

    @requires_dataset
    def convert_datetime_to_numerical(self, cols:list[str] = []):  
        self.dataset.clean_data(cols_date=cols)   
