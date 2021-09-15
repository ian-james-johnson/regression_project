import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import pearsonr, spearmanr, ttest_ind

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, explained_variance_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import RFE, SelectKBest, f_regression

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import acquire
import prepare



def get_tax_info(df):
    # This is the columns that will be used to predict tax rate by county
    tax_info = df[['taxamount', 'taxvaluedollarcnt', 'fips']]
    
    tax_info = tax_info.dropna()
    
    # This reduces our dataset to 1000 randomly selected rows
    # The original dataset has too many points to easily plot
    tax_info = tax_info.sample(n=1000, axis=0)
    
    # This is the tax rate
    tax_info['tax_rates'] = round((df.taxamount / df.taxvaluedollarcnt) * 100, 2)
    
    # Setting up tax rates by county from fips
    los_angeles = tax_info[tax_info.fips == 6037].tax_rates
    orange = tax_info[tax_info.fips == 6059].tax_rates
    ventura = tax_info[tax_info.fips == 6111].tax_rates
    
    # Now to plot the tax rates by county
    plt.figure(figsize = (10, 5))

    plt.subplot(311)
    plt.hist(los_angeles, bins = 100)
    plt.title('Los Angeles County Tax Rates')
    plt.xlabel('Tax Rates (%)')
    plt.ylabel('Frequency')

    plt.subplot(312)
    plt.hist(orange, bins = 100)
    plt.title('Orange County Tax Rates')
    plt.xlabel('Tax Rates (%)')
    plt.ylabel('Frequency')

    plt.subplot(313)
    plt.hist(ventura, bins = 100)
    plt.title('Ventura County Tax Rates')
    plt.xlabel('Tax Rates (%)')
    plt.ylabel('Frequency')

    plt.subplots_adjust(bottom=3, top=7)

    plt.show()
    
    return tax_info