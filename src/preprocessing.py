import pandas as pd
import numpy as np
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

def get_preprocessing_pipeline():
    # Define sub pipelines for numerical 
    # Handle missing values with KNNImputer and scale the data
    numerical_pipeline = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=5)),
        ('log', FunctionTransformer(np.log1p, validate=True, feature_names_out='one-to-one')),
        ('scaler', StandardScaler())])
    
    # Define sub pipelines for categorical
    # Handle missing values with KNNImputer and encode the data
    categorical_pipeline = Pipeline(steps=[
        ('encoder', OneHotEncoder(handle_unknown='ignore'))])
    
    #Combine the numerical and categorical pipelines into a single preprocessor
    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_pipeline, ['Age', 'Fare', 'SibSp', 'Parch']),
        ('categorical', categorical_pipeline, ['Sex', 'Embarked'])
    ], remainder='drop')
    
    return preprocessor