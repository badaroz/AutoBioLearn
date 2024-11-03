from typing_extensions import deprecated
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score,accuracy_score
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from AutoBioLearn import AutoBioLearn
from decorators import apply_per_grouping, requires_dataset
from helpers import ModelHelper
from imblearn.over_sampling import SMOTE

class AutoBioLearnClassification(AutoBioLearn):
    def __init__(self) -> None:
        self.__balancing = False
        super().__init__()

    def set_balancing(self, balancing:bool)-> None:
        self.__balancing = balancing
          
    @deprecated("Method will be deprecated, consider using execute_models")
    def run_models(self, models:list[str]=["xgboost"],  times_repeats:int=10, params={}, section:str=None):
        self.execute_models(models, times_repeats,params)

    @requires_dataset
    @apply_per_grouping    
    def execute_models(self, models:list[str]=["xgboost"],  times_repeats:int=10, params={}, section:str=None):       
        
        models_execution = {}
        if not self.data_processor.dataset.get_has_many_header():
            self._models_executed = []
        
        for model_name in models:
            models_execution[model_name] = ModelHelper.get_model(model_name, "classifier")      

        x = self.data_processor.dataset.get_X(section)
        try:
            y = self.data_processor.dataset.get_Y(section)
        except:
            y = self.data_processor.dataset.get_Y()

        for model_name, (model_object, model_params_hidden_verbosity) in models_execution.items():

            model_params = ModelHelper.get_model_params(model_name,params)       
            combination_params = ParameterGrid(model_params)
            
            for current_params in combination_params:
                
                for i in range(times_repeats):
                
                        for validation, validation_params in self._validations_execution.items():  
                           
                            ix_list = ModelHelper.initialize_validation(validation_params['validation'], \
                                                                        validation_params['num_folds'],  \
                                                                        validation_params['train_size'], \
                                                                        x, y)

                            for fold, (train_index, test_index) in enumerate(ix_list):
                                x_train = x.iloc[train_index]
                                y_train = y.iloc[train_index]
                                x_test = x.iloc[test_index]
                                y_test = y.iloc[test_index]
                                
                                if self.__balancing:
                                    x_train,y_train=SMOTE().fit_resample(x_train,y_train)

                                model_instance = model_object()
                                merged_params = {**current_params, **model_params_hidden_verbosity}

                                model_instance.set_params(**merged_params)
                                model_instance.fit(x_train, y_train)                                 

                                y_pred = model_instance.predict(x_test)

                                model_name_table = f'{model_name}_{str(current_params)}' if len(current_params) >0  else model_name
                                self._add_model_executed(i,validation, fold, model_name_table,model_instance,y_pred, y_test,test_index, section)    
                                    

    @deprecated("Method will be deprecated, consider using execute_models_with_best_model")
    def run_models_with_best_model(self, models:list[str]=["xgboost"],  
                                   times_repeats:int=10,
                                   params={}, 
                                   params_method="grid",
                                   section: str=None):
        
        self.execute_models_with_best_model(models,times_repeats,params, params_method,section)
        
    @requires_dataset
    @apply_per_grouping
    def execute_models_with_best_model(self, models:list[str]=["xgboost"],  
                                   times_repeats:int=10,
                                   params={}, 
                                   params_method="grid",
                                   section: str=None):
        models_execution = {}

        if params_method == 'random':
            model_gen =  RandomizedSearchCV
        else:
            model_gen = GridSearchCV

        for model_name in models:
            models_execution[model_name] = ModelHelper.get_model(model_name, "classifier")      

        x = self.data_processor.dataset.get_X(section)
        try:
            y = self.data_processor.dataset.get_Y(section)
        except:
            y = self.data_processor.dataset.get_Y()

        if not self.data_processor.dataset.get_has_many_header():
            self._models_executed = []

        for model_name, (model_object, model_params_hidden_verbosity) in models_execution.items():

            model_params = ModelHelper.get_model_params(model_name,params)       
            model_instance = model_object()
                          

            model_instance.set_params(**model_params_hidden_verbosity)
            best_params = self._find_best_hyperparams(model_instance, \
                                                    x, \
                                                    y, \
                                                    model_params, \
                                                    model_gen, \
                                                    ModelHelper.get_validations("split"),1,70)
            for i in range(times_repeats):                
                        for validation, validation_params in self._validations_execution.items():  
                           
                            ix_list =  ModelHelper.initialize_validation(validation_params['validation'],\
                                                                        validation_params['num_folds'],\
                                                                        validation_params['train_size'], \
                                                                        x,y)     

                            for fold, (train_index, test_index) in enumerate(ix_list):
                                x_train = x.iloc[train_index]
                                y_train = y.iloc[train_index]
                                x_test = x.iloc[test_index]
                                y_test = y.iloc[test_index]                               

                                model_instance = model_object()                             
                                merged_params = {**best_params, **model_params_hidden_verbosity}

                                model_instance.set_params(**merged_params)

                                model_instance.fit(x_train, y_train)                                 

                                y_pred = model_instance.predict(x_test)
                                
                                model_name_table = f'{model_name}_{str(best_params)}' if len(best_params) >0  else model_name
                                self._add_model_executed(i,validation, fold, model_name_table,model_instance,y_pred, y_test,test_index, section)     
                                
    @apply_per_grouping
    @deprecated("Method will be deprecated, consider using evaluate_models")
    def eval_models(self, metrics: list[str] = ["Recall","Precision","Accuracy","F1","ROC-AUC"], section: str = None) -> dict:
        return super().evaluate_models(metrics, section)
    
    @apply_per_grouping
    def evaluate_models(self, metrics: list[str] = ["Recall","Precision","Accuracy","F1","ROC-AUC"], section: str = None) -> dict:
        return super().evaluate_models(metrics, section)
    
    def _calculate_metrics(self):
        metrics = []
        for row in self._models_executed:
            y_test = row["y_test"]
            y_pred = row["y_pred"]
            if "section" in row:
                metrics.append((row["model_name"], row["section"], row["validation"],row["time"], row["fold"],\
                                                        precision_score(y_true= y_test,y_pred= y_pred), \
                                                        accuracy_score(y_true= y_test,y_pred= y_pred), \
                                                        recall_score(y_true= y_test,y_pred= y_pred), \
                                                        f1_score(y_true= y_test,y_pred= y_pred), \
                                                        roc_auc_score(y_true= y_test,y_score= y_pred)))
               
            else:
                metrics.append((row["model_name"], row["validation"],row["time"], row["fold"],\
                                                        precision_score(y_true= y_test,y_pred= y_pred), \
                                                        accuracy_score(y_true= y_test,y_pred= y_pred), \
                                                        recall_score(y_true= y_test,y_pred= y_pred), \
                                                        f1_score(y_true= y_test,y_pred= y_pred), \
                                                        roc_auc_score(y_true= y_test,y_score= y_pred)))
        
        if self.data_processor.dataset.get_has_many_header():
            cols_names = ["Model", "Section",
                            "Validation",\
                            "Time_of_execution",\
                            "Fold",\
                            "Precision","Accuracy",\
                            "Recall","F1","ROC-AUC"]
        else:
           cols_names = ["Model",\
                        "Validation",\
                        "Time_of_execution",\
                        "Fold",\
                        "Precision","Accuracy",\
                        "Recall","F1","ROC-AUC"]
      
        self._metrics = pd.DataFrame(data = metrics, columns=cols_names)
        
    @apply_per_grouping  
    def plot_metrics(self, metrics:list[str]=["Recall","Precision","Accuracy","F1","ROC-AUC"],rot=90, figsize=(12,6), fontsize=20,section: str = None):
       return super().plot_metrics(metrics = metrics,rot= rot,figsize= figsize, fontsize= fontsize, section= section)