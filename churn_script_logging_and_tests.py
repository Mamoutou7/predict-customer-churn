# library doc string
'''
Module for testing function from churn_library.py

Author : Mamoutou Fofana
Date: Jan 4, 2023
'''

import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import():
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        data_frame = cls.import_data("./data/bank_data.csv")
        logging.info('SUCCESS: import_data tested')

    except FileNotFoundError as err:
        logging.error('import_data tested: file not found')
        raise err
    try:
        assert data_frame.shape[0] > 0
        assert data_frame.shape[1] > 0
        logging.info('DataFrame shape -> Rows: %d\tColumns: %d',
                     data_frame.shape[0], data_frame.shape[1])
    except AssertionError as err:
        logging.error(
            "import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    data_frame = cls.import_data("./data/bank_data.csv")
    try:
        cls.perform_eda(data_frame)
        logging.info("SUCCESS: perfom_eda tested successfully")
    except AttributeError as err:
        logging.error(
            "perform_eda tested: input should be a dataframe")
        raise err
    # Figures list
    figure_lst = ["Churn_Distribution.png", "Customer_Age_Distribution.png",
                  "Marital_Status_Distribution.png",
                  "Total_Trans_Ct_Distribution.png', 'Heatmap.png"]

    for figure in figure_lst:
        try:
            assert os.path.isfile("./images/eda/" + figure) is True
            logging.info("SUCCESS: File %s was found", figure)
        except AssertionError as err:
            logging.error("Not such file on disk")
            raise err


def test_encoder_helper():
    '''
    test encoder helper
    '''

    cat_columns = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"
    ]
    data_frame = cls.import_data("./data/bank_data.csv")
    response = "Churn"

    try:
        cls.encoder_helper(
            data_frame=data_frame,
            category_lst=cat_columns,
            response=response)
        logging.info("SUCCESS: encoder_helper tested successfully")
    except KeyError as err:
        logging.error(
            "encoder_helper: Some column names doesn't exist in your dataframe")
        raise err
    try:
        assert isinstance(cat_columns, list)
        assert len(cat_columns) > 0

    except AssertionError as err:
        logging.error(
            "category_lst argument should be a list with length > 0")
        raise err


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    try:

        data_frame = cls.import_data("./data/bank_data.csv")
        target = "Churn"

        cls.perform_feature_engineering(data_frame, response=target)

        logging.info(
            "SUCCESS: perform_feature_engineering tested successfully")

    except KeyError as err:
        logging.error(
            "Target column names doesn't exist in dataframe")
        raise err

    try:
        assert isinstance(target, str)

    except AssertionError as err:
        logging.error(
            "response argument should be a string")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    dataframe = cls.import_data("./data/bank_data.csv")
    x_train, x_test, y_train, y_test = cls.perform_feature_engineering(
        data_frame=dataframe, response='Churn')
    try:
        cls.train_models(x_train, x_test, y_train, y_test)
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info("File %s was found', 'logistic_model.pkl")
    except MemoryError as err:
        logging.error(
            "Not such file on disk")
        raise err

    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info("File %s was found', 'logistic_model.pkl")
    except MemoryError as err:
        logging.error(
            "Not such file on disk")
        raise err


if __name__ == '__main__':

    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
