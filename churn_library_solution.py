# library doc string
'''
Predict Customer Churn Project with Clean Code Principles from the Machine Learning DevOps Nanodegree Program

Author: Mamoutou FOFANA

Date: December 16, 2022

release: 0.0.1
'''

import shap
import joblib
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
# linear algebra
import numpy as np
# For creating plots
import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import plot_roc_curve, classification_report


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
        # Checking that the df variable df has dataFrame type
        assert isinstance(df, pd.DataFrame)
        logging.info("SUCCESS: df is DataFrame type")
    except AssertionError:
        logging.error("ERROR: argument df in perform_eda must be {} but is {}".format(pd.DataFrame, type(df)))

    # A deep copy of the DataFrame (df)
    eda_df = df.copy()

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

    # Return dataframe
    return eda_df


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

    # Copy DataFrmae
    encoder_df = df.copy()

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

    # Category list to turn in new column
    category_lst = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]

    # Feature engineering (Verify if category_lst is empty)
    encoded_df = encoder_helper(df, category_lst, response)

    # target dataframe
    y = encoded_df["Churn"]

    # New dataframe
    X = pd.DataFrame()

    keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                 'Total_Relationship_Count', 'Months_Inactive_12_mon',
                 'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                 'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                 'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                 'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
                 'Income_Category_Churn', 'Card_Category_Churn']

    X[keep_cols] = encoded_df[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


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

    # RandomForestClassifier
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25,
             str('Random Forest Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6,
             str('Random Forest Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')

    # LogisticRegression
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25,
             str('Logistic Regression Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05,
             str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.text(0.01, 0.6,
             str('Logistic Regression Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test, y_test_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in path
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(output_pth + 'feature_importances.png')


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

    # Random Forest Classifier and Logistic Regression
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Grid Search parameters
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    # Grid Search fitting for Random Forest Classifier
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    # Train the Logistic Regression
    lrc.fit(X_train, y_train)

    # Save best models of the Random Forest Classifier and Logistic Regression
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Compute train and test predictions for Random Forest Classifier
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    # Compute train and test predictions for Logistic Regression
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)


    # Plotting the ROC curve for the logistic regression
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    lcr_plot = plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    rfc_plot = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    plt.show()

    # Classification report for training and testing results storing as image
    classification_report_image(y_train,
                          y_test,
                          y_train_preds_lr,
                          y_test_preds_lr,
                          y_train_preds_rf,
                          y_test_preds_rf)

    # Creating and storing of the feature importances
    feature_importance_plot(model=cv_rfc,
                            X_data=X_test,
                            output_pth='./images/results/')


if __name__ == '__main__':

    # import bank data
    df_path = import_data("./data/bank_data.csv")

    # perform Exploratory Data Analysis (EDA)
    eda_df = perform_eda(df=df_path)

    # perform features engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(df=eda_df, response="Churn")

    # Model training, prediction and evaluation
    train_models(X_train=X_train,
                 X_test=X_test,
                 y_train=y_train,
                 y_test=y_test)