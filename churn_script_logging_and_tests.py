"""
chrun_script_logging_and_tests.py

testing and logging script for churn_library.py.
Testing all functions and logging workflow errors/successes

Author: Stefan Lietz
Date: 19/4/2025
"""

import os
import logging
from churn_library import import_data, perform_eda, encoder_helper

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
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
		logging.info(f"Testing import_data: dataframe contains {df.shape[0]} rows and {df.shape[1]} columns")
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda():
	'''
	test perform eda function
	'''
	df = import_data("./data/bank_data.csv")
	if df.isnull().sum().sum() > 0:
		logging.warning(f"test_eda: dataframe contains {df.isnull().sum().sum()} NaN values")
	else:
		logging.info("test_eda: SUCCESS, no NaN values")

	perform_eda(df)
	# get all file names in folder images/eda:
	images = os.listdir("./images/eda")
	for file in ['countplot_marital_status.png', 'dist_total_transaction.png', 'heatmap_all_cols.png', 'hist_churn.png', 'hist_customer_age.png']:
		try:
			assert file in images
			logging.info(f"test_eda: {file} found in images/eda")
		except AssertionError:
			logging.error(f"test_eda: {file} does not exist")


def test_encoder_helper():
	'''
	test encoder helper
	'''
	df = import_data("./data/bank_data.csv")
	try:
		df_encode = encoder_helper(df)
	except:
		Print("xx")


def test_perform_feature_engineering(perform_feature_engineering):
	'''
	test perform_feature_engineering
	'''


def test_train_models(train_models):
	'''
	test train_models
	'''


if __name__ == "__main__":
	test_import()
	test_eda()
	test_encoder_helper()








