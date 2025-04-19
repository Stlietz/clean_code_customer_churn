# library doc string
"""
churn_library.py

completes the process for solving the data science process to idenfity customer churn including:
- EDA
- Feature Engineering (including encoding of categorical variables)
- Model Training
- Prediction
- Model Evaluation

Author: Stefan Lietz
Date: 19/4/2025
"""


# import libraries
import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report

os.environ['QT_QPA_PLATFORM']='offscreen' # For matplotlib and seaborn - Don’t try to open any graphical window — just render off-screen



def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(fr"{pth}")
    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            eda_df: pandas dataframe with new churn column
    '''
    # Make deep copy of originally loaded DataFrame
    eda_df = df.copy(deep=True)

    # Column for Customer Churn
    eda_df['Churn'] = eda_df['Attrition_Flag'].apply(lambda val: 0 if val=="Existing Customer" else 1)

    #  Histogram of Churn
    plt.figure(figsize=(15, 8))
    eda_df['Churn'].hist()
    plt.savefig(fname='./images/eda/hist_churn.png')

    # Histogram for Customer Age
    plt.figure(figsize=(15, 8))
    eda_df['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/hist_customer_age.png')

    # Countplot Marital Status
    plt.figure(figsize=(15, 8))
    eda_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/countplot_marital_status.png')

    # Distribution of total transaction
    plt.figure(figsize=(15, 8))
    sns.histplot(eda_df['Total_Trans_Ct'],kde=True);
    plt.savefig(fname='./images/eda/dist_total_transaction.png')

    # Heatmap all columns
    plt.figure(figsize=(20, 10))
    sns.heatmap(eda_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap_all_cols.png')

    # Return dataframe
    return eda_df



def encoder_helper(df, category_lst=['Gender','Education_Level','Marital_Status','Income_Category','Card_Category'], response="Churn"):

	'''
	helper function to turn each categorical column into a new column with
	propotion of churn for each category - associated with cell 15 from the notebook

	input:
			df: pandas dataframe
			category_lst: list of columns that contain categorical features
			response: string of response name [optional argument that could be used for naming variables or index y column]

	output:
			df: pandas dataframe with new columns for further proceeding
	'''
	df_encode = df.copy(deep=True)
	for cat_col in category_lst:
		churn_map = dict(df.groupby(cat_col).mean()['Churn'])
		df[cat_col + response] = df[cat_col].map(churn_map)
  
	return df_encode



def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    pass


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    pass

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    pass



if __name__ == "__main__":
    # import data
    df = import_data("./data/bank_data.csv")
    eda_df = perform_eda(df)
    df_encode = encoder_helper(eda_df)
    print(df.head())