"""
chrun_script_logging_and_tests.py

testing and logging script for churn_library.py.
Testing all functions and logging workflow errors/successes

Author: Stefan Lietz
Date: 19/4/2025
"""

import os
import logging
import pandas as pd
# , train_models, feature_importance_plot, classification_report_image
from churn_library import import_data, perform_eda, encoder_helper, perform_feature_engineering


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
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info(
            "Testing import_data: dataframe contains %d rows and %d columns",
            df.shape[0], df.shape[1])
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda():
    '''
    test perform eda function
    '''
    df = import_data("./data/bank_data.csv")
    df = perform_eda(df)
    if df.isnull().sum().sum() > 0:
        logging.warning(
            "test_eda: dataframe contains %d NaN values", df.isnull().sum().sum())
    else:
        logging.info("test_eda: No NaN values")
    # get all file names in folder images/eda:
    images = os.listdir("./images/eda")
    for file in [
        'countplot_marital_status.png',
        'dist_total_transaction.png',
        'heatmap_all_cols.png',
        'hist_churn.png',
            'hist_customer_age.png']:
        try:
            assert file in images
            logging.info("test_eda: SUCESS, %s found in images/eda", file)
        except AssertionError:
            logging.error("test_eda: %s does not exist", file)


def test_encoder_helper():
    '''
    test encoder helper
    '''
    df = import_data("./data/bank_data.csv")
    df = perform_eda(df)
    df_encode = encoder_helper(df)
    try:
        df_encode = encoder_helper(df)
        logging.info("test_encoder_helper: SUCCESS, encoding ran")
        if df_encode.shape[1] <= df.shape[1]:
            logging.warning(
                f"test_encoder_helper: encoded df has only {df_encode.shape[1]} columns,\
                    i.e. less or equal than original df with {df.shape[1]} columns")
    except Exception as err:
        logging.error("test_encoder_helper: encoding did not work")
        raise err
    try:
        pd.testing.assert_frame_equal(df, df_encode[df.columns])
    except AssertionError:
        logging.error(
            "encoded dataframe and original df are not equal w.r.t. original columns")
        raise AssertionError(
            "encoded dataframe and original df are not equal w.r.t. original columns")


def test_perform_feature_engineering():
    '''
    test perform_feature_engineering
    '''
    df = import_data("./data/bank_data.csv")
    df = perform_eda(df)
    try:
        x_train, x_test, y_train, y_test = perform_feature_engineering(df)
    except Exception as err:
        logging.error("test_perform_feature_engineering: %s", err)
    try:
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert y_train.shape[0] > 0
        assert y_test.shape[0] > 0
        logging.info(
            "test_perform_feature_engineering: SUCCESS, data split into train and test sets")
    except AssertionError as err:
        logging.error(
            "test_perform_feature_engineering: One or more files appear to not have rows and/or columns")
        raise err


def test_train_models():
    '''
    test train_models
    '''
    models = os.listdir("./models")
    for file in ['logistic_model.pkl', 'rfc_model.pkl']:
        try:
            assert file in models
            logging.info(
                "test_train_models: SUCCESS, %s found in models folder", file)
        except AssertionError:
            logging.error("test_train_models: %s does not exist", file)


def test_result_plots():
    '''
    test feature_importance_plot and classification_report_image
    '''
    images = os.listdir("./images/results")
    for file in [
        'lr_results.png',
        'rf_results.png',
        'feature_importance.png',
            'roc_curve.png']:
        try:
            assert file in images
            logging.info(
                "test_reult_plots: SUCCESS, %s found in images/results", file)
        except AssertionError:
            logging.error("test_reult_plots: %s does not exist", file)


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
    test_result_plots()