# library doc string
'''
Predict Customer Churn Project

Author: Mamoutou Fofana
Date: December 16, 2022
'''

# import libraries
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import joblib
# data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd
# linear algebra
import numpy as np
# For creating plots
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

warnings.simplefilter("ignore")


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''

    # load the dataset from csv file path
    data_frame = pd.read_csv(pth)

    # Create new feature called Churn
    data_frame['Churn'] = data_frame['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Drop columns
    data_frame = data_frame.drop(['Unnamed: 0', 'CLIENTNUM', 'Attrition_Flag'], axis=1)
    return data_frame


def perform_eda(data_frame):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Numeric data
    num_data = ['Churn', 'Customer_Age', 'Total_Trans_Ct']
    # Plot and save the Churn, Customer_Age and Total_Trans_Ct distribution
    for column in data_frame[num_data]:
        plt.figure(figsize=(20, 10))
        data_frame[column].hist()
        print(f"[INFO] Create Histogram plot of {column}")
        plt.title(f"{column} Distribution")
        plt.savefig(fname=f"./images/eda/{column}_distribution.png")

    # Plot and save the Marital Status distribution
    print("[INFO] Create Histogram plot of Marital_Status")
    plt.figure(figsize=(20, 10))
    data_frame.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_distribution.png')
    # Plot and save the Heatmap
    print("[INFO] Create Heatmap plot")
    plt.figure(figsize=(20, 10))
    sns.heatmap(data_frame.corr(), annot=False,
                cmap='Dark2_r', linewidths=2)
    plt.savefig('./images/eda/heatmap.png')


def encoder_helper(data_frame, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''

    # Copy DataFrame
    encoder_df = data_frame.copy()

    for category in category_lst:
        column_lst = []
        column_groups = data_frame.groupby(category).mean()['Churn']
        for val in data_frame[category]:
            column_lst.append(column_groups.loc[val])
        if response:
            encoder_df[category + '_' + response] = column_lst
        else:
            encoder_df[category] = column_lst

    return encoder_df


def perform_feature_engineering(data_frame, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be
              used for naming variables or index y column]

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
    # Turn the above categorical column list into new columns encoded
    encoded_df = encoder_helper(data_frame, category_lst, response)

    # target dataframe
    target = encoded_df["Churn"]

    # New dataframe
    predictor_data = pd.DataFrame()

    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    predictor_data[keep_cols] = encoded_df[keep_cols]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(
        predictor_data, target, test_size=0.3, random_state=42)

    return x_train, x_test, y_train, y_test


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
    # Random forest classifier report
    plt.rc('figure', figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/rf_results.png')
    plt.close()
    # Logistic regression report
    plt.rc('figure', figsize=(5, 5))

    plt.text(0.01, 1.25, str('Logistic Regression Train'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test,
                                                  y_test_preds_lr)),
             {'fontsize': 10},
             fontproperties='monospace')
    plt.axis('off')
    plt.savefig(fname='./images/results/logistic_results.png')
    plt.close()


def feature_importance_plot(model, x_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''

    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]
    # Create plot
    plt.figure(figsize=(20, 8))
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')
    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    # Save the image
    plt.savefig(fname=f"{output_pth}/feature_importance.png")
    plt.close()


def train_models(x_train, x_test, y_train, y_test):
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
    grid_search = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    # Train the Logistic Regression
    lrc.fit(x_train, y_train)

    # Save best the models of the Random Forest Classifier and Logistic Regression
    joblib.dump(grid_search.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Plot the ROC curve for the logistic regression
    plt.figure(figsize=(15, 8))
    plot_roc_curve(lrc, x_test, y_test,
                   ax=plt.gca(), alpha=0.8)
    plot_roc_curve(grid_search.best_estimator_,
                   x_test,
                   y_test,
                   ax=plt.gca(),
                   alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve_result.png')
    plt.close()


if __name__ == '__main__':
    # dataframe path
    DF_PATH = "./data/bank_data.csv"

    # import bank data
    DATA_FRAME = import_data(pth=DF_PATH)

    perform_eda(data_frame=DATA_FRAME)

    # perform features engineering
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = perform_feature_engineering(
        data_frame=DATA_FRAME,
        response="Churn")

    # Model training
    train_models(x_train=X_TRAIN,
                 x_test=X_TEST,
                 y_train=Y_TRAIN,
                 y_test=Y_TEST
                 )

    # Models loading
    RFC = joblib.load('./models/rfc_model.pkl')
    LR = joblib.load('./models/logistic_model.pkl')

    # Model prediction and evaluation
    Y_TRAIN_PREDS_LR = LR.predict(X_TRAIN)
    Y_TRAIN_PREDS_RF = RFC.predict(X_TRAIN)
    Y_TEST_PREDS_LR = LR.predict(X_TEST)
    Y_TEST_PREDS_RF = RFC.predict(X_TEST)

    # Classification report for training and testing results storing as image
    classification_report_image(y_train=Y_TRAIN,
                                y_test=Y_TEST,
                                y_train_preds_lr=Y_TRAIN_PREDS_LR,
                                y_train_preds_rf=Y_TRAIN_PREDS_RF,
                                y_test_preds_lr=Y_TEST_PREDS_LR,
                                y_test_preds_rf=Y_TEST_PREDS_RF
                                )

    X_DATA = pd.concat([X_TRAIN, X_TEST])
    OUTPUT_PTH = './images/results'

    # Creating and storing of the features important
    feature_importance_plot(model=RFC, x_data=X_DATA, output_pth=OUTPUT_PTH)
