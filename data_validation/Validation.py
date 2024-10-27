from data_treatement.Dataset import Dataset
from sklearn.model_selection import train_test_split

class Validator:
    def __init__(self, dataset: Dataset, X, Y, validations:list[str]=["stratified_kfold"], train:float = 70.0) -> None:
        
        if Dataset is None and (X is None or  Y is None):
            raise AttributeError("Dataset or X and Y must have a value")
        
        X = dataset.get_X() if X is None  else X
        Y = dataset.get_Y() if Y is None else Y

        self.__Validations = validations     
        self.__X_train, self.__X_test, self.__Y_train, self.__Y_test = train_test_split(X,Y, train_size=(train/100),stratify=Y)

    
    def get_validations(self)->list[str]:
        return self.__Validations
    
    def set_validations(self, validations:list[str]):
        self.__Validations = validations

    #def get