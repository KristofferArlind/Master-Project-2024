import pandas as pd
import numpy as np
import sys

def process_data(data, target_variable, horizon, weekday = None, comp_id = "gvkey", date_shift = False):
    
    data = data.copy()

    sys.stdout.write("Processing data\n")
    sys.stdout.write("Target variable: " + target_variable + "\n")
    sys.stdout.write("Horizon: " + str(horizon) + "\n")
    sys.stdout.write("Comp id: " + comp_id + "\n")
    sys.stdout.write("Date shift: " + str(date_shift) + "\n")
    sys.stdout.write("Data shape: " + str(data.shape) + "\n")
    sys.stdout.write("Data columns: " + str(data.columns) + "\n")

    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")

    fwd_variable = f'{target_variable}_fwd'
    
    if (date_shift):
        final_features_df = pd.read_csv('final_features.csv', delimiter=';')
    else:
        final_features_df = pd.read_csv('final_features_n.csv', delimiter=';')
    final_features = final_features_df[final_features_df['final_feature'].notnull()]["final_feature"].to_list()

    final_features = [feature for feature in final_features if feature in data.columns]
    final_features_df = final_features_df[final_features_df["final_feature"].isin(final_features)]

    data = data[final_features]

    dtype_dict = dict(zip(final_features_df["final_feature"], final_features_df["feature_type"]))
    del dtype_dict['date']
    
    for index, feature in final_features_df.iterrows():
        if pd.isnull(feature["final_feature"]) or feature["final_feature"] == "date":
            continue
        if feature["feature_type"] == "string":
            data[feature["final_feature"]] = data[feature["final_feature"]].astype(str)
        elif feature["feature_type"] == "int" and feature["categorical"] == True:
            data[feature["final_feature"]] = data[feature["final_feature"]].astype(str)
        elif feature["feature_type"] == "int":
            data[feature["final_feature"]] = data[feature["final_feature"]].astype(int)
        elif feature["feature_type"] == "float":
            data[feature["final_feature"]] = data[feature["final_feature"]].astype(float)
        if feature["categorical"] == True:
            data[feature["final_feature"]] = data[feature["final_feature"]].astype("category")
            
    if date_shift:
        data.set_index("date", inplace=True)
        data[fwd_variable] = data.groupby(comp_id)[target_variable].transform(lambda x: x.shift(-horizon, freq="D"))
        data.reset_index(inplace=True)
    else:
        data[fwd_variable] = data.groupby([comp_id])[target_variable].shift(-horizon)

    max_date = data['date'].unique().max()
    min_date = data['date'].unique().min()
    data = data[(data["date"] > min_date) & (data["date"] < max_date)]

    data.dropna(subset=[fwd_variable], inplace=True)
    
    data[fwd_variable] = data[fwd_variable].astype(float)

    if weekday != None:
        data = data[data["date"].dt.weekday == weekday]

    return data

def create_train_validation_test(data, test_date, validation=0.2, horizon_days=5, random_validation_dates=False):
    data_test = None
    data_validation = None

    data["date"] = pd.to_datetime(data["date"], format="%Y-%m-%d")
    all_dates = data["date"].unique()

    if (test_date != None):
        test_date = pd.to_datetime(test_date, format="%Y-%m-%d")

        all_dates = (all_dates[all_dates < test_date])

        data_test = data[data["date"] == test_date]
        if len(data_test) == 0:
            print("Test date not in dataset")
            return None, None, None

        all_dates = all_dates[:-horizon_days]

    if (random_validation_dates):
        np.random.shuffle(all_dates)
        
    training_days, validation_days = np.split(all_dates, [int(len(all_dates)* (1-validation))])
    
    if validation > 0:
        training_days = training_days[:-horizon_days]
        data_validation = data[data["date"].isin(validation_days)]
    
    data_train = data[data["date"].isin(training_days)]
    
    return data_train, data_validation, data_test

from scipy.stats.mstats import winsorize
def winsorize_data(df, columns=None, upper_pc = 0.99, lower_pc = 0.01):
    
    df = df.copy()
    if columns == None:
        columns = df.columns
    if "date" not in df.columns:
        for column in columns:
            df[column] = winsorize(df[column], limits=[lower_pc, 1 - upper_pc])
        return df
    for column in columns:
        if column == "date":
            continue
        df[column] = df.groupby("date")[column].transform(lambda x: winsorize(x, limits=[lower_pc, 1 - upper_pc])).reindex(level=1)
    if "date" in df.columns:
        df.drop(columns="date", inplace=True)
    return df

def blocked_time_series_split(df, embargo_weeks = 1, months_per_block = 6, val_weeks = 4):
    dates_df = pd.DataFrame(columns=["train_start", "train_end", "val_start", "val_end"])
    index_results = []
    
    df["date"] = pd.to_datetime(df["date"])
    
    min_date = df["date"].min()
    max_date = df["date"].max()
    current_date = min_date
    
    while (current_date + pd.DateOffset(weeks=months_per_block*4)) < max_date:
        train_start = current_date
        val_end = current_date + pd.DateOffset(weeks=months_per_block*4)
        val_start = val_end - pd.DateOffset(weeks=val_weeks)
        train_end = val_start - pd.DateOffset(weeks=embargo_weeks)
        
        dates_df = pd.concat([dates_df, pd.DataFrame({"train_start": train_start, "train_end": train_end, "val_start": val_start, "val_end": [val_end]})], ignore_index=True)
        
        index_results.append((np.array(df[df["date"].between(train_start, train_end)].index), np.array(df[df["date"].between(val_start, val_end)].index)))
                
        current_date = val_end + pd.DateOffset(weeks=1)
        
    if max_date - val_end > pd.Timedelta(weeks=int(0.5 * months_per_block*4)):
        train_start = current_date
        val_end = max_date
        val_start = val_end - pd.DateOffset(weeks=val_weeks)
        train_end = val_start - pd.DateOffset(weeks=embargo_weeks)
        index_results.append((np.array(df[df["date"].between(train_start, train_end)].index), np.array(df[df["date"].between(val_start, val_end)].index)))
        
    return index_results, dates_df
    
def print_size_of_splits(df, cv_split):
    total_train_size = 0
    total_val_size = 0
    
    max_train_size = 0
    max_val_size = 0
    
    sys.stdout.write("Number of splits: " + str(len(cv_split)) + "\n")
    sys.stdout.write("Size of all data: " + str(df.memory_usage(deep=True).sum()/(1024**2)) + "\n")
    sys.stdout.write("Shape of all data: " + str(df.shape) + "\n")

    for split in cv_split:
        total_train_size += df.iloc[split[0]].memory_usage(deep=True).sum()/(1024**2)
        total_val_size += df.iloc[split[1]].memory_usage(deep=True).sum()/(1024**2)
        max_train_size = max(max_train_size, df.iloc[split[0]].memory_usage(deep=True).sum()/(1024**2))
        max_val_size = max(max_val_size, df.iloc[split[1]].memory_usage(deep=True).sum()/(1024**2))

    sys.stdout.write("Total train size (MB): " + str(total_train_size) + "\n")
    sys.stdout.write("Avg train size (MB): " + str(total_val_size/len(cv_split)) + "\n")
    sys.stdout.write("Max train size (MB): " + str(max_train_size) + "\n")
    sys.stdout.write("Total val size (MB): " + str(total_val_size) + "\n")
    sys.stdout.write("Avg val size (MB): " + str(total_val_size/len(cv_split)) + "\n")
    sys.stdout.write("Max val size (MB): " + str(max_val_size) + "\n")
    