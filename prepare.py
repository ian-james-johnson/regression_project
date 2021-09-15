import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import  MinMaxScaler

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns

import acquire
import explore

def prepare_zillow(df):
    '''
    This prepares zillow df for analysis.
    Only the features needed for MVP are kept.
    Columns are given more meaningfull names.
    Nulls are dropped.
    Properties without bedrooms or bathrooms are dropped.
    '''
    
    # Get dataframe with only features needed for the MVP
    zillow = df

    # Give column more meaningful names
    zillow = zillow.rename(columns={'calculatedfinishedsquarefeet':'area', 
                                    'bedroomcnt':'bedroom', 
                                    'bathroomcnt':'bathroom', 
                                    'taxvaluedollarcnt':'taxvalue'})

    # Drop rows with null values
    zillow = zillow.dropna()

    # Drop properties with no bedrooms or bathrooms
    zillow = zillow[zillow.bedroom > 0]
    zillow = zillow[zillow.bathroom > 0]
    
    return zillow



def zillow_split(zillow):
    '''
    This function splits the df into train, validate, and test subsets.
    '''
    # Split the data into train, validate, test, subsets
    train_validate, test = train_test_split(zillow, test_size=.2, random_state=123)
    train, validate = train_test_split(zillow, test_size=.25, random_state=123)

    return train, validate, test



def zillow_scale(x_train, x_validate, x_test):
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




#XXXXXXXXXXXX older version
def zillow_scaleXXX(train, validate, test):
    '''
    This function scales data after it has been split into train, validate, and test subsets.
    '''
    
    # Create subsets with only predictive features (x)
    # Create subsets with only target feature (y)
    x_train = train.drop(columns='taxvalue')
    y_train = train.taxvalue
    x_validate = train.drop(columns='taxvalue')
    y_validate = train.taxvalue
    x_test = train.drop(columns='taxvalue')
    y_test = train.taxvalue


    # Create the scaler object
    scaler = MinMaxScaler()

    # Fit the scaler
    scaler.fit(x_train)

    # Use the scaler
    train_scaled = scaler.transform(x_train)
    validate_scaled = scaler.transform(x_validate)
    test_scaled = scaler.transform(x_test)

    return train_scaled, validate_scaled, test_scaled

