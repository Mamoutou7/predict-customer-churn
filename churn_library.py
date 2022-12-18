# library doc string
'''
Predict Customer Churn Project with Clean Code Principles from the Machine Learning DevOps Nanodegree Program

Author: Mamoutou FOFANA

Date: December 16, 2022

release: 0.0.1
'''

# linear algebra
import numpy as np
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
from datetime import date
# For creating plots
import matplotlib.pyplot as plt
import seaborn as sns


import logging
import os
os.environ['QT_QPA_PLATFORM']='offscreen'


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
sns.set(style='white')


logging.basicConfig(
    filename='./logs/churn_library.log',
    filemode='w',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'    
)

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    
    try:
        # load the dataset from csv file path
        df = pd.read_csv(pth)
        logging.info("SUCCESS: There are {} rows in your dataframe ".format(df.shape))
        return df 
    except FileNotFoundError:
        logging.error("ERROR: We are not able to find file {} ".format(pth))

def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    try:
        # Verifying that the df variable df has dataFrame type
        assert isinstance(df, pd.DataFrame)
        logging.info("SUCCESS: df is DataFrame type")
    except AssertionError:
        logging.error("ERROR: argument df in perform_eda must be {} but is {}".format(pd.DataFrame, type(df)))

    # A deep copy of the DataFrame (df)
    eda_df = df.copy(deep=True)

    # Computing and plotting the Churn distribution
    eda_df['Churn'] = eda_df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)

    # Plotting the Churn distribution
    plt.figure(figsize=(20, 10))
    eda_df['Churn'].hist()
    plt.savefig('./images/eda/churn_distribution.png')

    # Plotting the Customer Age distribution plotting
    plt.figure(figsize=(20, 10))
    eda_df['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_distribution.png')

    # Computing and plotting Marital Status distribution
    plt.figure(figsize=(20, 10))
    eda_df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_distribution.png')

    # Computing and plotting the Total Transaction Distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(eda_df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_transaction_distribution.png')

    # Heatmap plotting
    plt.figure(figsize=(20, 10))
    sns.heatmap(eda_df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    # A deep copy of the DataFrame (df)
    encoder_df = df.copy(deep=True)

    for category in category_lst:
        column_lst = []
        column_groups = df.groupby(category).mean()['Churn']
        for val in df[category]:
            column_lst.append(column_groups.loc[val])
        if response:
            encoder_df[category + '_' + response] = column_lst
        else:
            encoder_df[category] = column_lst

    return encoder_df



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


if __name__ == '__main__':
    # import bank data
    INITIAL_DF = import_data("./data/bank_data.csv")

    # perform Exploratory Data Analysis (EDA)
    #eda_df = perform_eda(df=INITIAL_DF)

    encoded_df = encoder_helper(df=INITIAL_DF, category_lst=['Gender', 'Education_Level'], response='Churn')
    print(encoded_df.head())

