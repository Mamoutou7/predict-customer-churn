# Predict Customer Churn

- Project **Predict Customer Churn** of Machine Learning DevOps Engineer Nanodegree Udacity

## Project Description

The project goal was to refactor a jupyter notebook (churn_notebook.ipynb) for building a customer churn prediction model by using clean code principles (best coding practices).

In this project, we identify credit card customers that are most likely to churn. The Project include a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested).

The package also have the flexibility of being run from the command-line interface (CLI).

## Files and data description

Overview of the files and data present in the root directory. 

```bash
.
├── Guide.ipynb          # Getting started and troubleshooting tips
├── churn_notebook.ipynb # Contains the code to be refactored
├── churn_library.py     # Different functions
├── churn_script_logging_and_tests.py # Tests and logs
├── README.md            # Project overview, and instructions to use the code
├── data                 # Read this data
│   └── bank_data.csv
├── images               # Store EDA results 
│   ├── eda
│   └── results
├── logs                 # Store logs
└── models               # Store models

```


## Dependencies

Here is a list of libraries used in this repository:

```
scikit-learn==1.1.3
joblib==1.1.1
pandas==1.5.2
numpy==1.23.4
matplotlib==3.6.2
seaborn==0.11.2
pylint==2.15.8
autopep8==2.0.0
pytest==7.1.2
```

To be able to run this project, you must install python library using the following command:

```
pip install -r requirements_py3-9.txt
```


## Running Files

Running the library file using the following command.

```
 python churn_library.py
```

The following steps executed:

- Loading the data.
- Encoding the categorical data.
- Creating EDA figures and saving then into the ./images/eda.
- Performing the feature engineering on the data.
- Train the Random Forest and Logistic Regression models and save them into the ./models file, then save the roc curves under ./images/results/.
- Find the feature importance if the random forest model and saving the yielded figure under ./images/results.



### Testing and Logging

`churn_script_logging_and_tests.py` enables to do testing and logging. Run below command:

```
python churn_script_logging_and_tests.py
```


### Cleaning up your code

Make sure the code you create complies with `PEP 8` rules. To check it automatically, run pylint on the terminal.

```
pylint churn_library.py
pylint churn_script_logging_and_tests.py
```

`Pylint` will provide recommendations for improvements in your code. A good code is a code that has a score close to 10.

To make repairs automatically, you can use autopep8.

```
autopep8 --in-place --aggressive --aggressive churn_script_logging_and_tests.py
autopep8 --in-place --aggressive --aggressive churn_library.py
```
