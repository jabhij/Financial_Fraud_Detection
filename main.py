# Importing python libraries- pandas & numpy
import pandas as pd
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D

# Importing python libraries- seaborn
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import average_precision_score
from xgboost.sklearn import XGBClassifier
from xgboost import plot_importance, to_graphviz

# Importing warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Checking the current directory
import os

cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)  # Get all the files in that directory
print("Files in %r: %s" % (cwd, files))

# Reading the csv file
df = pd.read_csv(r'....\Fraud.csv')
print(df.head())

# Checking for any missing values
df.isnull().values.any()


# Data Cleaning
X = df.loc[(df.type == 'CASH-IN') | (df.type == 'CASH-OUT') | (df.type == 'DEBIT') | (df.type == 'PAYMENT') | (df.type == 'TRANSFER')]

randomState = 5
np.random.seed(randomState)

Y = X['isFraud']
del X['isFraud']

# Eliminate columns shown to be irrelevant for analysis in the EDA
X = X.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis = 1)

# Binary-encoding of labelled data in 'type'
X.loc[X.type == 'TRANSFER', 'type'] = 0
X.loc[X.type == 'CASH_OUT', 'type'] = 1
X.type = X.type.astype(int) # convert dtype('O') to dtype(int)
