# Description

**AutoBioLearn**  is a package developed in Python. It is designed for development a pipeline easily of ML and explation of models executed. This package contains methods to analisys, treatment of dataset, training, validation, analisys metrics and results, and explations methods for all models executed.

Following environment preparation and usage examples, users will be able to reproduce results from their local computers.

Note - AutoBioLearn is still under development - we cannot guarantee full functionality

# Installation

To make it easier to configure the environment, a file has been made available with the libraries needed for requirements file, which can be found here
to install command `pip install -r requirements.txt`
# matplotlib compatibility
If run data_analisys before plot SHAP results and models metrics comparations, include command `%matplotlib inline` before import **AutoBioLearn** package, just like Example1.ipynb
# Examples
Examples can find in examples folder
# Dependency libraries
pandas==2.2.2
ydata-profiling==4.8.3
scikit-learn==1.5.1
xgboost==2.0.3
catboost==1.2.3
lightgbm==4.5.0
shap==0.42.1
imbalanced-learn==0.12.4
ipywidgets==8.1.2