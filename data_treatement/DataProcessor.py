from data_treatement import Dataset
from decorators.DatasetDecorators import requires_dataset
from helpers.ContentHelper import ContentHelper


class DataProcessor:
    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    @requires_dataset
    def encode_categorical(self, cols):
        self.dataset.convert_cols_values(cols)

    @requires_dataset
    def drop_cols_na(self, percent=30.0):
        self.dataset.drop_na(axis=ContentHelper.const_axis_column(), percent=percent, show_dropped=True)

    @requires_dataset
    def drop_rows_na(self, percent=10.0):        
        self.dataset.drop_na(axis=ContentHelper.const_axis_row(), percent=percent, show_dropped=True)
    
    @requires_dataset
    def show_cols_na(self):        
        self.dataset.print_na(axis=ContentHelper.const_axis_column())
    
    @requires_dataset
    def show_rows_na(self):        
        self.dataset.print_na(axis=ContentHelper.const_axis_row())

    @requires_dataset
    def remove_cols(self, cols:list[str] = [], cols_levels=0):      
        self.dataset.clean_data(cols_to_drop =cols, cols_levels= cols_levels)

    @requires_dataset
    def encode_numerical(self, cols:list[str] = [], cols_levels=0):  
        self.dataset.clean_data(cols_date=cols, cols_levels= cols_levels)   
