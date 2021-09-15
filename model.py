import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import RFE, SelectKBest, f_regression

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import acquire
#import prepare
#import explore



def zillow_scale(train, validate, test):
    '''
    This function scales data after it has been split into train, validate, and test subsets.
    '''
   
    # Create the scaler object
    scaler = MinMaxScaler()

    # Fit the scaler
    scaler.fit(x_train)

    # Use the scaler
    x_train_scaled = scaler.transform(x_train)
    x_validate_scaled = scaler.transform(x_validate)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_validate_scaled, x_test_scaled



def select_kbest(x_train_scaled, y_train, x_train, k):
    '''
    This function takes predictive features (x_train), target features (y_train), 
    and the number of top features (k) that we want to select.
    It returns the top predictive features that correlate with the target.
    '''
    
    # Create the object
    kbest = SelectKBest(f_regression, k)
    
    # Fit the object
    kbest = kbest.fit(x_train_scaled, y_train)

    # Convert back into a pd dataframe
    x_train_scaled = pd.DataFrame(x_train_scaled)

    # Return the columns names back to their original names (names were lost when kbest converted them into series)
    x_train_scaled.columns = x_train.columns

    # Use the object
    kbest = x_train.columns[kbest.get_support()]
    
    return kbest



def select_rfe(x_train_scaled, y_train, x_train, k):
    '''
    This function uses RFE to find the best predictive features
    for the target.
    '''
    
    rfe = RFE(estimator=LinearRegression(), n_features_to_select=k)
    rfe.fit(x_train_scaled, y_train)
    rfe.get_support()
    rfe_best = x_train.columns[rfe.get_support()]
    
    return rfe_best