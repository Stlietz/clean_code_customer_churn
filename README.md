# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project,implements learnings of the above mentioned course to identify credit card customers who are most likely to churn. The completed project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package can also be run interactively or from the command-line interface (CLI).

This project is intended to give practice using skills for testing and logging and using the best coding practices introduced in the respective lessons. On top of that, it also introduces the student to a problem data scientists across companies always face. How do we identify (and later intervene with) customers likely to churn?

## Files and data description
Overview of the files and data present in the root directory. 

**Folders:**
- data: contains bank_data.csv file
- images:
    - eda: output of exploratory data analysis
    - results: model results (confusion matrices, roc curves) and feature importance
- logs: log file which stores information created during testing
- models: saved logistic regression and random forest models as pkl files

**Main Project Files**
- churn_library.py: running from terminal executes the entire DS process, creating models and model results, saved in the above mentioned folders
- churn_notebook.ipynb: to iteratively run the project steps
- churn_script_logging_and_test.py: for testing and creating logs


## Running Files
1. Clone Repo
2. Create virtual environment with python 3.8
3. Install requirements using: python -m pip install -r requirements_py3.8.txt
4. Run: python churn_library.py python_script_logging_and_tests.py

*Note*: The project can as well be run in the respective jupyter notebook in interactive mode


