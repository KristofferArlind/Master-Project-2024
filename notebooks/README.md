# notebooks folder
These are the different notebookes which were run locally and/or on an IDUN cluster. Different testing notebooks were used to test various things during the entire thesis process, but most are not kept here as they are very messy and not complete as code has been replaced and removed in them over time.

## regime_data
Notebooks concerning the data used for training the regime models.

### different_weight_index.ipynb
Notebook used to get the index imitating the S&P 500 index. Naming comes from some attempts at weighting indices differntly.

### indicators_us.ipynb
Description of where all the raw US indicators were downloaded from, and formatting into a standard format.

### indicators_us_collected.ipynb
Collecting all the US indicators into one parquet file

## regime_model
Notebooks concerning the regime models.

### EPU.ipynb
Used to make the Economic Uncertainty Policy filter.

### lstm.ipynb
The LSTM models used to predict bear/bull markets, NBER recessions and the return of the index.

### markov_regression.ipynb
The notebook which processes data for the markov model ran in matlab, and using the probabilities to get the classified dates.

### qualitative.ipynb
The notebook which converts the qualitative bear/bull periods into specific dates.

### simple_filters.ipynb
The notebook used to make the return filters.

## results
The notebooks used to analyse and process different results.

### create_digests.ipynb
Notebook used to make smaller averages from the many stock model predictions to be able to anayse more effectively. Used to contain all digests, but most are moved to scripts. Whats left is the combining of ensemble models which are used to run t-tests.

### lookup_files.ipynb
To create files including some information on different companies which can be joined on the gvkeys and dates in the result files to get more information helpful for analysis.

### regime_results.ipynb
The analysis of in-sample regime classifications

### regime_results_oos.ipynb
The analysis of out-of-sample regime predictions

### stock_results.ipynb
The analysis of the stock results, as well as of the adaptive model

## stock_data
Notebooks regarding stock data

### after_cap_stats.ipynb
To plot the market cap and number of stocks in different regions after the market cap and volume cutoff.

### data_fund_annual.ipynb
Notebook for processing the annual fundamentals data before being joined on daily data in the scripts.

### handle_forex.ipynb
Processing of the forex data used to convert foreign values to USD.



