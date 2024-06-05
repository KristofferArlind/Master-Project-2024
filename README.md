# Master-Project-2024
Code used for my masters thesis. This code was all written by me, with syntax and help with bug-fixing from GitHub Copilot and Microsoft Edge Copilot. The code can be messy in certain places, as it was only written to be understood by me as well as obtain the results described in the thesis. If there are any discrepancies in the code compared to details in the thesis, the thesis takes precedence.

## Overview
The code and and some of the results data divided into 5 folders with the following contents. The folders have their own readme files with further explanations. A data folder is also referred to in some of the code. For copyright reasons this data could not be uploaded to a public GitHub repository. The process in obtaining the data is explained in the thesis.

### final_features_n_csv
csv file with information regarding the final features used for training stock models. Some of them may however be dropped, see the scripts in the model folder under scripts.

### final_fundamental_features_annual.csv
The final features denoted as fundamentals. Used to delay these features in the run stock model script.

### notebooks
Includes the different Jupyter notebooks which were used to analyze and process data, analyze and process results and run some of the regime models.

### results
Some of the results from the regime models are here. The raw results from the stock models are too large to store in a GitHub repository.

### scripts
Scripts which were ran on IDUN in order to process data and obtain results. Jobs which needed more power or took to long to run in Jupyter notebooks are placed here.

### slurms
Some of the different slurm job specification files which were used to run jobs on IDUN

### time_periods
The different defined time periods which were used to select data for training and testing.
