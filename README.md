# Predict Customer Churn

- Project **Predict Customer Churn** of Machine Learning DevOps Engineer Nanodegree Udacity

## Project Description

The project goal was to refactor a jupyter notebook (churn_notebook.ipynb) for building a customer churn prediction model by using clean code principles (best coding practices).
In the project, we identify credit card customers that are most likely to churn. 

## Files and data description
Overview of the files and data present in the root directory. 
```bash
./predict_customer_churn
                       |------data
                       |         |--- bank_data.csv
                       |------ images/
                       |          |--- eda/
                       |          |--- results/
                       |
                       |------ logs/
                       |------ models/
                       |------ churn_library.py
                       |------ churn_script_logging_and_tests.py
                       |------ LICENSE
                       |------ README.md
                       |------ requirements.txt
```

## Running Files
Running the library file using the following command.
```bash 
 python churn_library.py
```

The following steps executed:

- Loading the data.
- Encoding the categorical data.
- Creating EDA figures and saving then into the ./images/eda.
- Performing the feature engineering on the data.
- Train the Random Forest and Logistic Regression models and save them into the ./models file, then save the roc curves under ./images/results/.
- Find the feature importance if the random forest model and saving the yielded figure under ./images/results


