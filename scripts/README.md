# scripts folder
This folder is further divided into two folders.

## utils.py
Different utility functions used in different scripts.

# markov_reg_all_5yr.m
MATLAB script used to run the Markov regression models

## data
The scripts used to run on IDUN to process the stock data. The different files are explained below.

### start_data_process.py
This is the file that is actually ran on IDUN. This file takes in a few inputs and runs the {region}_data_n_pct.py files in batches in order to not run out of memory.

### {region}_data_n_pct.py
These files process the actual raw data which is downloaded from WRDS. The non-us data uses Compustat and the us data uses CRSP. All regions use Compustat fundamentals. The global_data_n_pct.py is not actually used in the thesis.

### stock_digests.py
This script takes all the different results from the stock model and creates smaller files showing the average return from the portfolios decided by the predictions, according to a few different input parameters.

### stock_digests_ensemble.py
This script does the same as stock_digests.py but for multiple models at the same time creating ensemble models.

### global_fundamental_features_annual.csv
Better names and other details about the fundamental features from Compustat. The fundamental data is processed in a notebook.

### us_fundamental_features_annual.csv
Same as global_fundamental_features_annual.csv but slightly different as the fundamental features available in Compustat is slightly different between NA and non-NA data.

## models
The scripts used to run on IDUN to run the different machine learning models on the stock data. The different files and folders are explained below.

### cv_values
The cross-validation values used for hyperparameter tuning the different stock models.

### objects
The different stock models objects created for easy use in the run_model scripts.

### choose_model.py
Simple script for getting a model object from a string input.

### run_model_n_tp.py
Script for running a a single algorithm on different kinds of data and with different train/test splits. All stock results described in the thesis is from models ran in this file





