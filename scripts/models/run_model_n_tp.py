import sys
sys.path.append("../scripts")
sys.path.append("../scripts/models")
import pandas as pd
pd.set_option("display.max_columns", 20)
import numpy as np
from sklearn.preprocessing import StandardScaler
import utils
from datetime import datetime
import os
import argparse
import uuid

from choose_model import choose_model

os.chdir("../")
parser = argparse.ArgumentParser()

parser.add_argument("-model", help="Model to run (xgb, catboost, logreg, svm, dnn, rf)", default="catboost")
parser.add_argument("--scale", help="Scale data", action="store_true")
parser.add_argument("-win_pc", help="Winsorization percentile to use", type=float, default=None)
parser.add_argument("--friday", help="Train only on fridays", action="store_true")
parser.add_argument("-region", help="Train on region (US, Europe, Japan)", default="US")
parser.add_argument("-first_train_year", help="First year to train on", type=int, default=1965)
parser.add_argument("-last_test_year", help="Last year to train on", type=int, default=2023)
parser.add_argument("-fundamentals_delay", help="Rows of fundamentals delay after fiscal period end", type=int, default=90)
parser.add_argument("--test_on_fridays", help="Test only on fridays", action="store_true")
parser.add_argument("-val_method", help="Method to use for validation (dates, blocked, time_series, random)", default=None)
parser.add_argument("-val_pc", help="Validation percentage of train data", type=float, default=0.0)
parser.add_argument("-val_date_csv", help="Path to csv with validation date periods", default=None)
parser.add_argument("--use_gpu", help="Use GPU for training", action="store_true")
parser.add_argument("-test_region", help="Region to test on (Global, US, Europe, Japan)", default=None)
parser.add_argument("-parallel", help="Number of parallel jobs to use", type=int, default=1)
parser.add_argument("--use_pct_trr", help="Use pct_change for all return rate", action="store_true")
parser.add_argument("-n_trials", help="Number of trials for hyperparameter optimization", type=int, default=50)
parser.add_argument("--no_mc_cap", help="Do not use market cap cutoff for train data", action="store_true")
parser.add_argument("-drop_cols", help="Drop these cols from training data col1,col2,col3", type=str, default=None)
parser.add_argument("-test_year_splits", help="Which years to train on data before and test on data after", type=str, default="2020,2015,2010,2005,2000,1995,1990,1985,1980")
parser.add_argument("-specific_train_files", help="Use only these train files (str in filename, sep by comma)", type=str, default=None)
parser.add_argument("-ignore_train_files", help="Ignore these train files (str in filename, sep by comma)", type=str, default=None)
parser.add_argument("--print_test_data", help="Print test data to output", action="store_true")

args, unknown = parser.parse_known_args()
args = vars(args)

hyperparameters = {}
for i in range(0, len(unknown), 2):
    key = unknown[i].lstrip('-')
    value = unknown[i + 1]
    hyperparameters[key] = value

sys.stdout.write(str(hyperparameters), "\n")

test_year_splits = args["test_year_splits"].split(",")

train_files = os.listdir("time_periods/model_train_ready")

train_files_before_test = os.listdir("time_periods/model_train_ready_before_test")

train_files = train_files + train_files_before_test

if args["specific_train_files"] != None:
    specific_train_files = args["specific_train_files"].split(",")
    new_train_files = []
    for specific_string in specific_train_files:
        new_train_files += [file for file in train_files if specific_string in file]
        
    train_files = new_train_files
    
elif args["ignore_train_files"] != None:
    ignore_train_files = args["ignore_train_files"].split(",")
    for specific_string in ignore_train_files:
        sys.stdout.write("Ignoring train files with string: " + specific_string + "\n")
        train_files = [file for file in train_files if specific_string not in file]
    
    
sys.stdout.write("Train files: " + str(train_files) + "\n")

for test_year in test_year_splits:
    test_year = int(test_year)
    for train_file in train_files:
        #OOS files, only use the ones trained until test_year
        if "train_" in train_file:
            if "train_" + str(test_year) not in train_file:
                continue

        if os.path.exists(f"time_periods/model_train_ready/{train_file}"):
            train_dates = pd.read_csv(f"time_periods/model_train_ready/{train_file}", delimiter=';')
            
        elif os.path.exists(f"time_periods/model_train_ready_before_test/{train_file}"):
            train_dates = pd.read_csv(f"time_periods/model_train_ready_before_test/{train_file}", delimiter=';')
            
        else:
            sys.stdout.write("Train file does not exists, continuing \n")
            continue
        
        train_dates["date"] = pd.to_datetime(train_dates["date"])
        
        test_dates = pd.DataFrame(list(pd.date_range(start=f'{test_year}-01-01', end=f'{args["last_test_year"]}-12-31', freq='D')), columns=["date"])
        
        train_dates["date"] = train_dates[train_dates["date"].dt.year < int(test_year)]["date"]
        
        if train_dates["date"].min() > pd.Timestamp(f"{test_year}-01-01"):
            print("Not early enough train dates for this file")
            continue


        sys.stdout.write("Now training on data: " + train_file + "\n")
        sys.stdout.write("Test year split: " + str(test_year) + "\n")
        
        model = choose_model(args["model"])

        model = model(hyperparameters, use_gpu=args["use_gpu"])

        run_time_and_date = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

        if args["test_region"] == None:
            args["test_region"] = args["region"]

        if args["val_method"] == "dates" and args["val_date_csv"] == None:
            raise Exception("Validation method dates specified but no validation date csv provided")

        if args["val_method"] == "random" and args["val_pc"] == 0:
            raise Exception("Validation method specified but validation percentage is 0")

        validation_percentage = args["val_pc"]

        test_on_fridays = args["test_on_fridays"]

        train_only_on_fridays = args["friday"]

        sys.stdout.write("Model name: " + str(args["model"]) + "\n")
        sys.stdout.write("Scale data: " + str(args["scale"]) + "\n")
        sys.stdout.write("Winsorization percentile: " + str(args["win_pc"]) + "\n")
        sys.stdout.write("Train only on fridays: " + str(train_only_on_fridays) + "\n")
        sys.stdout.write("Region to train on: " + str(args["region"]) + "\n")
        sys.stdout.write("Validation percentage: " + str(validation_percentage) + "\n")
        sys.stdout.write("Fundamentals delay: " + str(args["fundamentals_delay"]) + "\n")

        assert(args["region"] == "ROW" or args["region"] == "Global" or args["region"] == "US" or args["region"] == "Europe" or args["region"] == "Japan")

        volume_usd_5_min = 1000

        if args["no_mc_cap"]:
            min_market_cap_percentile_us = 0.0
            min_market_cap_percentile_row = 0.0
            min_market_cap_percentile_eu = 0.0
            min_market_cap_percentile_jp = 0.0
        else:
            min_market_cap_percentile_us = 0.6
            min_market_cap_percentile_row = 0.65
            min_market_cap_percentile_eu = 0.65
            min_market_cap_percentile_jp = 0.65

        target_horizon = 5 #Rows == 1 week == 5 business days
            
        target_variable = f'trr_{target_horizon}'

        final_features_df = pd.read_csv('final_features_n.csv', delimiter=';')

        final_features = final_features_df[final_features_df['final_feature'].notnull()]["final_feature"].to_list()

        first_train_year = int(args["first_train_year"])

        if (args["region"] != "US") and (first_train_year < 1992):
            first_train_year = 1992

        original_first_train_year = first_train_year
        if args["region"] == "US" and (first_train_year != 1965):
            first_train_year -= 1
        elif first_train_year != 1992:
            first_train_year -= 1
        last_train_year = test_year - 1

        train_years = list(range(first_train_year, last_train_year + 1))

        if args["region"] == "US":
            us_data_list = []
            for year in train_years:
                us_data_list.append(pd.read_parquet(f'data/processed/csrp/us_data_{year}_annual_fund_processed_n_pct.parquet', engine='pyarrow'))

            us_data = pd.concat(us_data_list)

            us_data["date"] = pd.to_datetime(us_data["date"], format='%Y-%m-%d')

            us_data["currency"] = "USD"
            us_data["country_hq"] = "US"

            us_data = utils.process_data(us_data, target_variable, target_horizon)
            sys.stdout.write("Number of gvkeys before cutoff: \n")
            for year in train_years:
                sys.stdout.write("Year: " + str(year))
                sys.stdout.write("Number of gvkeys: " + str(us_data[us_data["date"].dt.year == year]["gvkey"].nunique()) + "\n")

            us_data = us_data[us_data["volume_usd_5"] > volume_usd_5_min]
            us_data = us_data.groupby("date").apply(lambda x: x[x["market_cap_usd"] > x["market_cap_usd"].quantile(min_market_cap_percentile_us)]).reset_index(drop=True)

            sys.stdout.write("Number of gvkeys after cutoff: \n")
            for year in train_years:
                sys.stdout.write("Year: " + str(year))
                sys.stdout.write("Number of gvkeys: " + str(us_data[us_data["date"].dt.year == year]["gvkey"].nunique()) + "\n") 
            
            data = us_data

        elif args["region"] == "Europe":
            europe_data_list = []
            for year in train_years:
                europe_data_list.append(pd.read_parquet(f'data/processed/europe/europe_data_{year}_annual_fund_processed_n_pct.parquet', engine='pyarrow'))

            europe_data = pd.concat(europe_data_list)

            europe_data["date"] = pd.to_datetime(europe_data["date"], format='%Y-%m-%d')

            europe_data = utils.process_data(europe_data, target_variable, target_horizon)

            sys.stdout.write("Number of gvkeys before cutoff: \n")
            for year in train_years:
                sys.stdout.write("Year: " + str(year))
                sys.stdout.write("Number of gvkeys: " + str(europe_data[europe_data["date"].dt.year == year]["gvkey"].nunique()) + "\n")

            europe_data = europe_data[europe_data["volume_usd_5"] > volume_usd_5_min]
            europe_data = europe_data.groupby("date").apply(lambda x: x[x["market_cap_usd"] > x["market_cap_usd"].quantile(min_market_cap_percentile_eu)]).reset_index(drop=True)

            sys.stdout.write("Number of gvkeys after cutoff: \n")
            for year in train_years:
                sys.stdout.write("Year: " + str(year))
                sys.stdout.write("Number of gvkeys: " + str(europe_data[europe_data["date"].dt.year == year]["gvkey"].nunique()) + "\n")    

            data = europe_data

        elif args["region"] == "Japan":
            japan_data_list = []
            for year in train_years:
                japan_data_list.append(pd.read_parquet(f'data/processed/japan/japan_data_{year}_annual_fund_processed_n_pct.parquet', engine='pyarrow'))

            japan_data = pd.concat(japan_data_list)

            japan_data["date"] = pd.to_datetime(japan_data["date"], format='%Y-%m-%d')

            japan_data = utils.process_data(japan_data, target_variable, target_horizon)

            sys.stdout.write("Number of gvkeys before cutoff: \n")
            for year in train_years:
                sys.stdout.write("Year: " + str(year))
                sys.stdout.write("Number of gvkeys: " + str(japan_data[japan_data["date"].dt.year == year]["gvkey"].nunique()) + "\n")

            japan_data = japan_data[japan_data["volume_usd_5"] > volume_usd_5_min]
            japan_data = japan_data.groupby("date").apply(lambda x: x[x["market_cap_usd"] > x["market_cap_usd"].quantile(min_market_cap_percentile_jp)]).reset_index(drop=True)

            sys.stdout.write("Number of gvkeys after cutoff: \n")
            for year in train_years:
                sys.stdout.write("Year: " + str(year))
                sys.stdout.write("Number of gvkeys: " + str(japan_data[japan_data["date"].dt.year == year]["gvkey"].nunique()) + "\n")

            data = japan_data

        data.sort_values(by=['date', 'gvkey'], inplace=True)
        data.reset_index(drop=True, inplace=True)
            
        fundamentals_delay = int(args["fundamentals_delay"])

        final_fundamental_features_df = pd.read_csv('final_fundamental_features_annual.csv', delimiter=';')
        final_fundamental_features = final_fundamental_features_df[final_fundamental_features_df['final_feature'].notnull()]["final_feature"].to_list()

        for feature in final_fundamental_features:
            data[feature] = data.groupby('gvkey')[feature].shift(fundamentals_delay)

        if original_first_train_year != first_train_year:
            data = data[data["date"] >= pd.Timestamp(f"{original_first_train_year}-01-01")]
        else:
            data = data.groupby('gvkey').apply(lambda x: x.iloc[fundamentals_delay:]).reset_index(drop=True)

        data.sort_values(by=['date', 'gvkey'], inplace=True)
        data.reset_index(drop=True, inplace=True)

        filtered_data = data[data["date"].isin(train_dates["date"])]
        data = filtered_data

        sys.stdout.write("Min train date: " + str(data["date"].min()) + "\n")
        sys.stdout.write("Max train date: " + str(data["date"].max()) + "\n")

        data[f'{target_variable}_fwd_class'] = 0
        data.loc[data.groupby(['date'])[f'{target_variable}_fwd'].transform(lambda x: x <= x.quantile(0.3333)), f'{target_variable}_fwd_class'] = -1
        data.loc[data.groupby(['date'])[f'{target_variable}_fwd'].transform(lambda x: x >= x.quantile(0.6666)), f'{target_variable}_fwd_class'] = 1

        sys.stdout.write(str(data[f'{target_variable}_fwd_class'].value_counts()) + "\n")

        min_test_year = min(test_dates["date"]).year
        original_min_test_year = min_test_year
        max_test_year = max(test_dates["date"]).year

        if (args["test_region"] == "US") and (min_test_year != 1965):
            min_test_year -= 1
        elif min_test_year != 1992:
            min_test_year -= 1

        test_years = list(range(min_test_year, max_test_year + 1))

        if args["test_region"] == "US":
            test_data_list_us = []
            for year in test_years:
                test_data_list_us.append(pd.read_parquet(f'data/processed/csrp/us_data_{year}_annual_fund_processed_n_pct.parquet', engine='pyarrow'))
                    
            test_data_us = pd.concat(test_data_list_us)
            
            test_data_us["date"] = pd.to_datetime(test_data_us["date"], format='%Y-%m-%d')

            test_data_us["currency"] = "USD"
            test_data_us["country_hq"] = "US"

            test_data = test_data_us
            
        elif args["test_region"] == "Europe":
            test_data_list_europe = []
            for year in test_years:
                test_data_list_europe.append(pd.read_parquet(f'data/processed/europe/europe_data_{year}_annual_fund_processed_n_pct.parquet', engine='pyarrow'))
                    
            test_data_europe = pd.concat(test_data_list_europe)
            
            test_data_europe["date"] = pd.to_datetime(test_data_europe["date"], format='%Y-%m-%d')

            test_data = test_data_europe

        elif args["test_region"] == "Japan":
            test_data_list_japan = []
            for year in test_years:
                test_data_list_japan.append(pd.read_parquet(f'data/processed/japan/japan_data_{year}_annual_fund_processed_n_pct.parquet', engine='pyarrow'))
                    
            test_data_japan = pd.concat(test_data_list_japan)
            
            test_data_japan["date"] = pd.to_datetime(test_data_japan["date"], format='%Y-%m-%d')

            test_data = test_data_japan
            
        elif args["test_region"] == "Global":
            test_data_list_us = []
            test_data_list_global = []
            for year in test_years:
                if year >= 1992:
                    test_data_list_global.append(pd.read_parquet(f'data/processed/global_data_{year}_annual_fund_processed_n_pct.parquet', engine='pyarrow'))
                test_data_list_us.append(pd.read_parquet(f'data/processed/csrp/us_data_{year}_annual_fund_processed_n_pct.parquet', engine='pyarrow'))

            test_data_us = pd.concat(test_data_list_us)
            test_data_global = pd.concat(test_data_list_global)
            
            test_data_us["date"] = pd.to_datetime(test_data_us["date"], format='%Y-%m-%d')
            test_data_global["date"] = pd.to_datetime(test_data_global["date"], format='%Y-%m-%d')
            
            test_data_us["currency"] = "USD"
            test_data_us["country_hq"] = "US"
            
            test_data = pd.concat([test_data_us, test_data_global])

        test_data.sort_values(by=['date', 'gvkey'], inplace=True)
        test_data.reset_index(drop=True, inplace=True)

        test_data = utils.process_data(test_data, target_variable, target_horizon)

        final_fundamental_features_df = pd.read_csv('final_fundamental_features_annual.csv', delimiter=';')
        final_fundamental_features = final_fundamental_features_df[final_fundamental_features_df['final_feature'].notnull()]["final_feature"].to_list()

        for feature in final_fundamental_features:
            test_data[feature] = test_data.groupby('gvkey')[feature].shift(fundamentals_delay)

        if original_min_test_year != min_test_year:
            test_data = test_data[test_data["date"] >= pd.Timestamp(f"{original_min_test_year}-01-01")]
        else:
            test_data = test_data.groupby('gvkey').apply(lambda x: x.iloc[fundamentals_delay:]).reset_index(drop=True)

        test_data.sort_values(by=['date', 'gvkey'], inplace=True)
        test_data.reset_index(drop=True, inplace=True)
        
        filtered_test_data = test_data[test_data["date"].isin(test_dates["date"])]
        detached_test_data = filtered_test_data.copy()

        detached_test_data[f'{target_variable}_fwd_class'] = 0
        detached_test_data.loc[detached_test_data.groupby(['date'])[f'{target_variable}_fwd'].transform(lambda x: x <= x.quantile(0.3333)), f'{target_variable}_fwd_class'] = -1
        detached_test_data.loc[detached_test_data.groupby(['date'])[f'{target_variable}_fwd'].transform(lambda x: x >= x.quantile(0.6666)), f'{target_variable}_fwd_class'] = 1

        sys.stdout.write(str(detached_test_data[f'{target_variable}_fwd_class'].value_counts()) + "\n")

        if train_only_on_fridays:
            data = data[data["date"].dt.weekday == 4]

        result_cols = ["date", "gvkey", "company_name", "currency", "exchange_code", target_variable + "_fwd", target_variable + "_fwd_class", "market_cap_usd"]

        if detached_test_data.columns.tolist() != data.columns.tolist():
            sys.stdout.write("Columns in detached test data and data are not the same\n")
            sys.stdout.write("Columns in detached test data: " + str(detached_test_data.columns.tolist()) + "\n")
            sys.stdout.write("Columns in data: " + str(data.columns.tolist()) + "\n")
            sys.stdout.write("Difference: " + str(list(set(detached_test_data.columns).difference(data.columns))) + "\n")
            common_cols = list(set(detached_test_data.columns).intersection(data.columns))
            detached_test_data = detached_test_data[common_cols]
            data = data[common_cols]
        
        drop_cols = ["gvkey", "company_name", "business_description", "city", "state", "price_close_usd", target_variable + "_fwd", target_variable + "_fwd_class", "date"]
        if args["region"] == "US":
            drop_cols = ["permco", "permno", "gvkey", "company_name", "price_close_usd", target_variable + "_fwd", target_variable + "_fwd_class", "date"]

        if args["drop_cols"] != None:
            drop_cols += args["drop_cols"].split(",")

        sys.stdout.write("Drop cols: " + str(drop_cols) + "\n")
        df_results = pd.DataFrame(columns=result_cols + ["pred_0", "pred_1", "pred_2", "pred_class"])

        if test_on_fridays:
            test_dates = detached_test_data[detached_test_data["date"].dt.weekday == 4]["date"].unique()
        else:
            test_dates = detached_test_data["date"].unique()

        test_dates = sorted(test_dates)

        test_first_date = min(test_dates)
        test_last_date = max(test_dates)

        data = data[data["date"] < test_first_date]

        data_train, data_validation, data_test = utils.create_train_validation_test(data, test_date=None, validation=validation_percentage, horizon_days=target_horizon)
                    
        X_train = data_train.copy()

        categorical_features = final_features_df[final_features_df["categorical"] == True]["final_feature"].to_list()
            
        categorical_features = [x for x in categorical_features if x not in drop_cols]
        categorical_features = [x for x in categorical_features if x in X_train.columns]

        numeric_features = X_train.columns.difference(categorical_features)
        numeric_features = [x for x in numeric_features if x not in drop_cols]
        numeric_features = [x for x in numeric_features if x in X_train.columns]
        
        sys.stdout.write("Categorical features: " + str(categorical_features) + "\n")
        sys.stdout.write("Numeric features: " + str(numeric_features) + "\n")

        for col in categorical_features:
            if X_train[col].isnull().any():
                X_train[col] = X_train[col].cat.add_categories("Unknown").fillna("Unknown")
                
        sys.stdout.write("Train data before winsorize and nanfill: " + str(X_train.head(50)) + "\n")

        for col in X_train.columns:
            if col in numeric_features:
                max_value = X_train[col].replace([np.inf], np.nan).max()
                min_value = X_train[col].replace([-np.inf], np.nan).min()
                X_train[col] = X_train[col].replace([np.inf], max_value)
                X_train[col] = X_train[col].replace([-np.inf], min_value)
                if args["win_pc"] != None:
                    X_train[col] = utils.winsorize_data(X_train[["date", col]], upper_pc=(1-args["win_pc"]), lower_pc=args["win_pc"])
                X_train[col] = X_train[col].fillna(0)

        sys.stdout.write("Columns: " + str(data_train.columns) + "\n")
        sys.stdout.write("Train data before scale: " + str(X_train.head(50)) + "\n")

        if args["scale"]:
            scaler = StandardScaler()
            scaler.fit(X_train[numeric_features])
            X_train[numeric_features] = scaler.transform(X_train[numeric_features])
            sys.stdout.write("Train data after scale: " + str(X_train.head(50)) + "\n")

        if args["val_method"] == "blocked":
            X_train.reset_index(drop=True, inplace=True)
            cv_split, cv_dates = utils.blocked_time_series_split(X_train.copy(), months_per_block=24, val_weeks=6*4)
            utils.print_size_of_splits(X_train, cv_split)
            X_train = X_train.drop(columns=drop_cols, errors="ignore")
            y_train = data_train[target_variable + "_fwd_class"]
            best_params_cv, cv_results_df = model.optuna_search(X = X_train, y = y_train, cv_indicies = cv_split, param_ranges = None, cat_features=categorical_features, par_jobs=args["parallel"], n_trials=args["n_trials"])

            model.fit(X_train, y_train, cat_features=categorical_features)
            

        elif validation_percentage > 0:
            X_train = X_train.drop(columns=drop_cols, errors="ignore")
            y_train = data_train[target_variable + "_fwd_class"]
            X_val = data_validation.copy()

            for col in categorical_features:
                if X_val[col].isnull().any():
                    X_val[col] = X_val[col].cat.add_categories("Unknown").fillna("Unknown")
            for col in X_val.columns:
                if col in numeric_features:
                    max_value = X_train[col].replace([np.inf], np.nan).max()
                    min_value = X_train[col].replace([-np.inf], np.nan).min()
                    X_val[col] = X_val[col].replace([np.inf], max_value)
                    X_val[col] = X_val[col].replace([-np.inf], min_value)
                    if args["win_pc"] != None:
                        X_val[col] = utils.winsorize_data(X_val[["date", col]], upper_pc=(1-args["win_pc"]), lower_pc=args["win_pc"])
                    X_val[col] = X_val[col].fillna(0)
                    
            X_val = X_val.drop(columns=drop_cols, errors="ignore")
            y_val = data_validation[target_variable + "_fwd_class"]

            model.fit(X_train, y_train, X_val, y_val, cat_features=categorical_features)

        else:
            X_train = X_train.drop(columns=drop_cols, errors="ignore")
            y_train = data_train[target_variable + "_fwd_class"]
            model.fit(X_train, y_train, cat_features=categorical_features)

        model.set_new_feature_importance(test_first_date, features=X_train.columns)
        
        sys.stdout.write(str(test_dates) + "\n")
        for test_date in test_dates:
            if args["print_test_data"]:
                sys.stdout.write("Test date: " + str(test_date) + "\n")
            
            data_test = detached_test_data[detached_test_data["date"] == test_date]
            
            if data_test.empty:
                continue

            X_test = data_test.drop(columns=drop_cols, errors="ignore")
            y_test = data_test[target_variable + "_fwd_class"]
            test_actual = data_test.copy().loc[:, [x for x in result_cols if x in data_test.columns]]
            if args["print_test_data"]:
                sys.stdout.write("Test data before winsorize and nanfill: " + str(X_test.head(10)) + "\n")
                sys.stdout.write("Test columns: " + str(X_test.columns) + "\n")

            for col in categorical_features:
                if X_test[col].isnull().any():
                    X_test[col] = X_test[col].cat.add_categories("Unknown").fillna("Unknown")
            for col in X_test.columns:
                if col in numeric_features:
                    max_value = X_train[col].replace([np.inf], np.nan).max()
                    min_value = X_train[col].replace([-np.inf], np.nan).min()
                    X_test[col] = X_test[col].replace([np.inf], max_value)
                    X_test[col] = X_test[col].replace([-np.inf], min_value)
                    if args["win_pc"] != None:
                        X_test[col] = utils.winsorize_data(X_test[[col]], upper_pc=(1-args["win_pc"]), lower_pc=args["win_pc"])
                    X_test[col] = X_test[col].fillna(0)
                
            if args["print_test_data"]:
                sys.stdout.write("Test data before scale: " + str(X_test.head(10)) + "\n")
                sys.stdout.write("Test columns: " + str(X_test.columns) + "\n")
            if args["scale"]:
                
                X_test_scaled = X_test.copy()
                X_test_scaled[numeric_features] = scaler.transform(X_test_scaled[numeric_features])
                if args["print_test_data"]:
                    sys.stdout.write("Test data after scale: " + str(X_test_scaled.head(10)) + "\n")
                    sys.stdout.write("Test columns: " + str(X_test_scaled.columns) + "\n")

                preds_proba = model.predict_proba(X_test_scaled)
                preds_class = model.predict(X_test_scaled)
            else:
                preds_proba = model.predict_proba(X_test)
                preds_class = model.predict(X_test)

            test_actual["pred_0"] = preds_proba[:, 0]
            test_actual["pred_1"] = preds_proba[:, 1]
            test_actual["pred_2"] = preds_proba[:, 2]
            test_actual["pred_class"] = preds_class
            
            df_results = pd.concat([df_results, test_actual])

        df_importances = model.get_feature_importances()
            
        df_results["pred_0"] = pd.to_numeric(df_results["pred_0"])
        df_results["pred_1"] = pd.to_numeric(df_results["pred_1"])
        df_results["pred_2"] = pd.to_numeric(df_results["pred_2"])
        df_results[target_variable + "_fwd"] = pd.to_numeric(df_results[target_variable + "_fwd"])
        
        df_results["market_cap_rank"] = df_results.groupby("date")["market_cap_usd"].rank(ascending=False, method="first").astype(int)
        df_results.drop(columns="market_cap_usd", inplace=True)

        model_name = str(args["model"]) + "_" + str(args["region"]) + "_" + run_time_and_date

        model_name = model_name + "_test_split_" + str(test_year) 

        model_name = model_name + "_train_file_" + str(train_file).split(".")[0][0:30]
        
        model_name = model_name + "_" + str(uuid.uuid4())[0:4]
        
        model_info = model_name

        if train_only_on_fridays:
            model_info = model_info + "_friday_trained"

        model_info = model_info + "_val_" + str(args["val_method"])

        model_info = model_info + "_win_pc_" + str(args["win_pc"])

        if args["scale"]:
            model_info = model_info + "_scaled"
            
        model_info = model_info + args["test_region"] + "_test"

        model_info = model_info + "_first_train_year_" + str(first_train_year) + "_last_train_year_" + str(last_train_year)

        model_info = model_info + "_min_vol_5_" 

        if args["region"] == "US":
            model_info = model_info + "_min_mcap_pct_us_" + str(min_market_cap_percentile_us)
        if args["region"] == "Europe":
            model_info = model_info + "_min_mcap_pct_eu_" + str(min_market_cap_percentile_eu)
        if args["region"] == "Japan":
            model_info = model_info + "_min_mcap_pct_jp_" + str(min_market_cap_percentile_jp)

        if not os.path.exists(f"results/{model_name}"):
            os.makedirs(f"results/{model_name}")

        hyperparameters_all = model.get_params()
        with open(f"results/{model_name}/hyperparameters.txt", 'w') as f:
            for key, value in hyperparameters_all.items():
                f.write('%s:%s\n' % (key, value))
                
        with open(f"results/{model_name}/model_info.txt", 'w') as f:
            f.write(model_info)

        df_results.to_parquet(f"results/{model_name}/results.parquet", index=False)
 
        mean_importances = model.get_mean_feature_importance()
        mean_importances.to_csv(f"results/{model_name}/importances.csv")

        train_dates.to_csv(f"results/{model_name}/{train_file}", index=False)
        train_dates.to_csv(f"results/{model_name}/train_dates.csv", index=False)
            
        if args["val_method"] != None:
            cv_dates.to_csv(f"results/{model_name}/cv_dates.csv", index=False)
            cv_results_df.to_csv(f"results/{model_name}/cv_results.csv", index=False)
            pd.DataFrame.from_dict(best_params_cv, orient="index").to_csv(f"results/{model_name}/cv_best_params.csv", index=False)

        df_importances.to_csv(f"results/{model_name}/all_importances.csv")

        model.save_model(f"results/{model_name}/")




