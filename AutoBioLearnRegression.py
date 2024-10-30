import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, root_mean_squared_error, r2_score, median_absolute_error
from sklearn.model_selection import GridSearchCV, ParameterGrid, RandomizedSearchCV
from AutoBioLearn import AutoBioLearn
from decorators.DatasetDecorators import apply_per_grouping, requires_dataset
from helpers import ModelHelper


class AutoBioLearnRegression(AutoBioLearn):
    @requires_dataset
    @apply_per_grouping
    def run_models(self, models:list[str]=["xgboost"],  times_repeats:int=10, params={}, section: str=None):

        models_execution = {}
        self._models_executed = []

        x = self.data_processor.dataset.get_X(section)
        try:
            y = self.data_processor.dataset.get_Y(section)
        except:
            y = self.data_processor.dataset.get_Y()


        for model_name in models:
            models_execution[model_name] = ModelHelper.get_model(model_name, "regressor")
            
        for model_name, (model_object, model_params_hidden_verbosity) in models_execution.items():

            model_params = ModelHelper.get_model_params(model_name,params)       
            combination_params = ParameterGrid(model_params)
            
            for current_params in combination_params:
                
                for i in range(times_repeats):
                
                        for validation, validation_params in self._validations_execution.items():  
                           
                            ix_list = ModelHelper.initialize_validation(validation_params['validation'],    \
                                                                        validation_params['num_folds'],     \
                                                                        validation_params['train_size'],    \
                                                                        x,y)

                            for fold, (train_index, test_index) in enumerate(ix_list):
                                x_train = x.iloc[train_index]
                                y_train = y.iloc[train_index]
                                x_test = x.iloc[test_index]
                                y_test = y.iloc[test_index]
                                
                              
                                model_instance = model_object()
                                merged_params = {**current_params, **model_params_hidden_verbosity}

                                model_instance.set_params(**merged_params)
                                model_instance.fit(x_train, y_train)                                 

                                y_pred = model_instance.predict(x_test)

                                model_name_table = f'{model_name}_{str(current_params)}' if len(current_params) >0  else model_name
                              
                                self._models_executed.append({"time":i,
                                                            "validation":validation,
                                                            "fold":fold,                                                           
                                                            "model_name":model_name_table,
                                                            "model":model_instance,
                                                            "y_pred":y_pred,
                                                            "y_test":y_test,
                                                            "x_test_index":test_index })
                                
    @requires_dataset
    @apply_per_grouping
    def run_models_with_best_model(self, models:list[str]=["xgboost"],  times_repeats:int=10,params={}, params_method="grid", section: str = None):
        models_execution = {}
        self._models_executed = []
        
        if params_method == 'random':
            model_gen =  RandomizedSearchCV
        else:
            model_gen = GridSearchCV

        for model_name in models:
            models_execution[model_name] = ModelHelper.get_model(model_name, "regressor")

        x = self.data_processor.dataset.get_X(section)
        try:
            y = self.data_processor.dataset.get_Y(section)
        except:
            y = self.data_processor.dataset.get_Y()

        for model_name, (model_object, model_params_hidden_verbosity) in models_execution.items():

            model_params = ModelHelper.get_model_params(model_name,params)       
            model_instance = model_object()
                          

            model_instance.set_params(**model_params_hidden_verbosity)
            best_params = self._find_best_hyperparams(model_instance,x, y,model_params,model_gen,ModelHelper.get_validations("split"),1,70)
            for i in range(times_repeats):                
                        for validation, validation_params in self._validations_execution.items():  
                           
                            ix_list = ModelHelper.initialize_validation(validation_params['validation'], \
                                                                        validation_params['num_folds'],  \
                                                                        validation_params['train_size'], \
                                                                        x, \
                                                                        y)

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
                                self._models_executed.append({"time":i,
                                                            "validation":validation,
                                                            "fold":fold,                                                           
                                                            "model_name":model_name_table,
                                                            "model":model_instance,
                                                            "y_pred":y_pred,
                                                            "y_test":y_test,
                                                            "x_test_index":test_index })

        
    def eval_models(self, metrics: list[str] = ["MSE","RMSE","R2","MAE","MAPE"]) -> dict:
        return super().eval_models(metrics)

    def _calculate_metrics(self):
        metrics = []
        for row in self._models_executed:
            y_test = row["y_test"]
            y_pred = row["y_pred"]
            metrics.append((row["model_name"], row["validation"],row["time"], row["fold"],
                                                    mean_squared_error(y_true= y_test,y_pred= y_pred), \
                                                    root_mean_squared_error(y_true= y_test,y_pred= y_pred), \
                                                    r2_score(y_true= y_test,y_pred= y_pred), \
                                                    median_absolute_error(y_true= y_test,y_pred= y_pred), \
                                                    mean_absolute_percentage_error(y_true= y_test,y_pred= y_pred)))
      
        self._metrics = pd.DataFrame(data = metrics, columns=["Model", \
                                                            "Validation", \
                                                            "Time_of_execution", \
                                                            "Fold", \
                                                            "MSE", \
                                                            "RMSE", \
                                                            "R2","MAE","MAPE"])
    
    def plot_metrics(self, metrics:list[str]=["MSE","RMSE","R2","MAE","MAPE"],rot=90, figsize=(12,6), fontsize=20):
       return super().plot_metrics(metrics = metrics,rot= rot,figsize= figsize, fontsize= fontsize)
    
