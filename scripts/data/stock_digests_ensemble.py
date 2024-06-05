
import argparse
import sys
import os

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("-regions", default="US,Europe,Japan")
parser.add_argument("-years",  default="1980,1985,1990,1995,2000,2005,2010,2015,2020")
parser.add_argument("-models", default="catboost,xgb,logreg,rf")
parser.add_argument("-min_trr_5", type=float, default=-0.3)
parser.add_argument("-max_trr_5", type=float, default=0.3)
parser.add_argument("--no_conviction", action="store_true")

args = parser.parse_args()
args = vars(args)

os.chdir("../")
sys.stdout.write("Current working directory: " + os.getcwd() + "\n")

def filter_market_caps(results_df, min_market_cap_percentile = 0.6, max_market_cap_percentile = None):
    current_results = results_df.copy()
    
    min_market_caps = current_results.groupby("date")["market_cap_usd"].quantile(min_market_cap_percentile)
    
    if max_market_cap_percentile != None:
        max_market_caps = current_results.groupby("date")["market_cap_usd"].quantile(max_market_cap_percentile)
           
    current_results = current_results.groupby("date").apply(lambda x: x[x["market_cap_usd"] >= min_market_caps.loc[x.name]]).reset_index(drop=True)

    if max_market_cap_percentile != None:
        current_results = current_results.groupby("date").apply(lambda x: x[x["market_cap_usd"] <= max_market_caps.loc[x.name]]).reset_index(drop=True)
    current_results.sort_values(["date", "gvkey"], inplace=True)

    return current_results.copy()

def add_quantiles(results_df, quantiles=10):
    results_df = results_df.copy()
    
    def g(df):
        df['conviction_quantile'] = pd.qcut(df['conviction'], quantiles, labels=False, duplicates="drop")
        df['top_quantile'] = pd.qcut(df['pred_2'], quantiles, labels=False, duplicates="drop")
        df['bottom_quantile'] = pd.qcut(df['pred_0'], quantiles, labels=False, duplicates="drop")
        return df
        
    results_df = results_df.groupby("date").apply(g).reset_index(drop=True)
    return results_df.copy()

def set_time_period(results_df, first_date, last_date):
    current_results = results_df.copy()
    current_results = current_results[current_results["date"] > pd.Timestamp(first_date)]
    current_results = current_results[current_results["date"] < pd.Timestamp(last_date)]
    return current_results.copy()

def prepare_results(df, exchange_codes = None, currencies = None, quantiles = 20, 
                    min_date = "2020-01-01", max_date="2023-12-31", n_gvkeys = 500, svm=False,
                    min_market_cap_percentile = 0.6,
                   use_percentile_cap = False, min_volume_usd_5 = 1000, lower_rank = None, max_market_cap_percentile=None):
    
    df = df.copy()
    

    if svm:
        df["conviction"] = df["pred_1"]
    else:
        df["conviction"] = df["pred_2"] - df["pred_0"]
        
    
    if exchange_codes != None:
        df = df[df["exchange_code"].isin(exchange_codes)]
    if currencies != None:
        df = df[df["currency"].isin(currencies)]
        
        
    df["trr_5_fwd_ar"] = np.exp(df["trr_5_fwd"]) - 1
    if use_percentile_cap:
        df = filter_market_caps(df, min_market_cap_percentile, max_market_cap_percentile)
    else:
        if "market_cap_usd" in df.columns:
            df = df[df["volume_usd_5"] >= min_volume_usd_5]
            df["market_cap_rank"] = df.groupby("date")["market_cap_usd"].rank(ascending=False, method="first").astype(int)
            df = df[df["market_cap_rank"] <= n_gvkeys]
            if lower_rank != None:
                df = df[df["market_cap_rank"] >= lower_rank]

        elif "market_cap_rank" in df.columns:
            df = df[df["volume_usd_5"] >= min_volume_usd_5]
            df = df[df["market_cap_rank"] <= n_gvkeys]
        else:
            sys.stdout.write("No market cap or rank in df")
    df = add_quantiles(df, quantiles=quantiles)
    df = set_time_period(df, min_date, max_date)
    
    return df

#Ensemble model
models = args["models"].split(",")
regions = args["regions"].split(",")

if args["no_conviction"]:
    top_quantile_name = "top_quantile"
    bottom_quantile_name = "bottom_quantile"
    
else:
    top_quantile_name = "conviction_quantile"
    bottom_quantile_name = "conviction_quantile"

n_quantiles = 40

quantiles_n = [1,2,4,8,50]

group_features = ["all", "gsector", "ggroup", "exchange_code"]

mean_features = ["trr_5_fwd_ar", "volume_usd_5", "volatility_5"]

qt_mc_ranges = [
    (0.89, 1.00), #500 in 2023
    (0.78, 1.00), #1000 in 2023
    (0.67, 1.00), #1500 in 2023
    (0.56, 1.00), #2000 in 2023
    (0.34, 1.00), #3000 in 2023
    (0.12, 1.00), #4000 in 2023
    (0.00, 1.00), #4500 in 2023
    (0.78, 0.89), #1000 to 501,
    (0.56, 0.78), #2000 to 1001,
    (0.34, 0.56), #3000 to 2001,
    (0.12, 0.34), #4000 to 3001,
    (0.00, 0.12), #5000 to 4001,
]

max_trr_5_fwd = float(args["max_trr_5"]) #Arithmetic = +100%
min_trr_5_fwd = float(args["min_trr_5"]) #Arithmetic = -50%



top_quantiles = [list(range(n_quantiles - 1, n_quantiles-n - 1,-1)) for n in quantiles_n]

if args["no_conviction"]:
    bottom_quantiles = top_quantiles
else:
    bottom_quantiles = [list(range(0,n,1)) for n in quantiles_n]

only_first_5_test_years = True

keep_uuids = False

years = [int(year) for year in args["years"].split(",")]

result_dirs = os.listdir("results")

us_lookup = pd.read_parquet("data/lookup/us_lookup.parquet", engine="pyarrow")
us_lookup["date"] = pd.to_datetime(us_lookup["date"])
us_lookup.set_index(["date", "gvkey"], inplace=True)

eu_lookup = pd.read_parquet("data/lookup/eu_lookup.parquet", engine="pyarrow")
eu_lookup["date"] = pd.to_datetime(eu_lookup["date"])
eu_lookup.set_index(["date", "gvkey"], inplace=True)

jp_lookup = pd.read_parquet("data/lookup/jp_lookup.parquet", engine="pyarrow")
jp_lookup["date"] = pd.to_datetime(jp_lookup["date"])
jp_lookup.set_index(["date", "gvkey"], inplace=True)

result_cols = ['date', 'gvkey', 'exchange_code',
       'trr_5_fwd', 'trr_5_fwd_class', 'pred_0', 'pred_1', 'pred_2',
        'train_file', 'split_year', 'gsector',
       'ggroup', 'gind', 'gsubind', 'market_cap_usd', 'trr_5', 'volume_usd_5', 'volatility_5']

train_files = ['return_filter_bear_m_short_2_3', 'bull_dates_sp500', 'markov_rec',
       'flat_dates_sp500', 'nber_recession_dates',
       'return_filter_bull_m_short_2_3', 'non_bear_dates_sp500',
       'nber_expansion_dates', 'bear_dates_sp500',
       'return_filter_bull_m_long_3_6_', 'all_dates',
       'return_filter_bear_m_long_3_6_', 'markov_exp', 'EPU_exp_2yr',
       'EPU_rec_2yr']

mean_columns = ["pred_0", "pred_1", "pred_2"]

for region in regions:
    for y_i, year in enumerate(years):
        first_results = True
        if year < 2005:
            if region == "Europe" or region == "Japan":
                continue
        sys.stdout.write("year:", year, "\n")
        for train_file in train_files:
            sys.stdout.write(f"Processing {region} for {train_file} in {year}\n")
            models_found = 0
            for m_1, model_name in enumerate(models):
                for directory in result_dirs:
                    if model_name + "_" + region in directory and "regime_feature" not in directory and f"test_split_{year}" in directory and train_file in directory: 
                        if train_file == "bear_dates_sp500" and "non_bear_dates_sp500" in directory:
                            continue
                        sys.stdout.write(f"Found {model_name} for {region} for {train_file} in {year}\n")
                        models_found += 1
                        sys.stdout.write("Models found: " + str(models_found) + "\n")
                        current_single_model_results = pd.read_parquet(f"results/{directory}/results.parquet", engine="pyarrow")
                        current_single_model_results["date"] = pd.to_datetime(current_single_model_results["date"])
                        current_single_model_results = current_single_model_results[current_single_model_results["date"] <= (pd.Timestamp(f"{year}-01-01") + pd.DateOffset(years=5))]
                        if m_1 == 0:
                            ensemble_results = current_single_model_results.copy()
                            continue
                        ensemble_results = pd.concat([ensemble_results, current_single_model_results])
            if models_found == len(models):
                current_results = ensemble_results.groupby(["date", "gvkey"])[mean_columns + ["trr_5_fwd", "trr_5_fwd_class"]].mean().reset_index()
                
                current_results["train_file"] = train_file
                current_results["split_year"] = year

                if "model" in current_results.columns:
                    current_results.drop(columns=["model"], inplace=True)

                if only_first_5_test_years:
                    current_results = current_results[current_results["date"] < (pd.Timestamp(f"{year}-01-01") + pd.DateOffset(years=5))]
                    
                if region == "US":
                    current_results = current_results.set_index(["date", "gvkey"]).merge(us_lookup, left_index=True, right_index=True, suffixes=("_x", "")).reset_index()
                elif region == "Europe":
                    current_results = current_results.set_index(["date", "gvkey"]).merge(eu_lookup, left_index=True, right_index=True, suffixes=("_x", "")).reset_index()
                elif region == "Japan":
                    current_results = current_results.set_index(["date", "gvkey"]).merge(jp_lookup, left_index=True, right_index=True, suffixes=("_x", "")).reset_index()
                    
                if max_trr_5_fwd:
                    current_results = current_results[current_results["trr_5_fwd"] <= max_trr_5_fwd]
                    
                if min_trr_5_fwd:
                    current_results = current_results[current_results["trr_5_fwd"] >= min_trr_5_fwd]

                current_results = current_results[result_cols]

                current_results_copy = current_results.copy()

                min_date = f"{year}-01-01"

                for n_i, min_max_quantiles in enumerate(qt_mc_ranges):
                    sys.stdout.write("quantiles:", min_max_quantiles, "\n")
                    current_results = current_results_copy.copy()

                    min_quantile = min_max_quantiles[0]
                    max_quantile = min_max_quantiles[1]
                    

                    current_results = prepare_results(current_results, quantiles=n_quantiles, min_date = min_date, 
                                                    use_percentile_cap=True, min_market_cap_percentile = min_quantile, 
                                                    max_market_cap_percentile=max_quantile)

                    for g_i, group_feature in enumerate(group_features):
                        sys.stdout.write("group_feature:", group_feature, "\n")
                        for q_i, quantile in enumerate(quantiles_n):
                            if group_feature == "all":
                                if q_i == 0:
                                    current_group_result_digest = pd.DataFrame(current_results["date"].unique(), columns=["date"]).set_index("date")
                                    current_group_result_digest["all"] = True

                                for mean_feature in mean_features:
                                    current_results_top_n = current_results[current_results[top_quantile_name].isin(top_quantiles[q_i])].groupby(["date"])[mean_feature].mean()
                                    current_results_bottom_n = current_results[current_results[bottom_quantile_name].isin(bottom_quantiles[q_i])].groupby(["date"])[mean_feature].mean()
                                    current_group_result_digest[f"top_{quantile}_{mean_feature}_mean"] = current_results_top_n
                                    current_group_result_digest[f"bottom_{quantile}_{mean_feature}_mean"] = current_results_bottom_n
                                    current_group_result_digest[f"top_minus_bottom_{quantile}_{mean_feature}_mean"] = current_results_top_n - current_results_bottom_n
                                    current_group_result_digest[f"mean_{quantile}_{mean_feature}_mean"] = (current_results_top_n + current_results_bottom_n)/2
                                    current_group_result_digest[f"top_{quantile}_n_stocks_chosen"] = current_results[current_results[top_quantile_name].isin(top_quantiles[q_i])].groupby(["date"])["gvkey"].count()
                                    current_group_result_digest[f"bottom_{quantile}_n_stocks_chosen"] = current_results[current_results[bottom_quantile_name].isin(bottom_quantiles[q_i])].groupby(["date"])["gvkey"].count()
                            else:
                                if q_i == 0:
                                    current_group_result_digest = current_results.groupby(["date", group_feature])["trr_5_fwd_ar"].mean().reset_index()[["date", group_feature]].set_index(["date", group_feature])

                                for mean_feature in mean_features:
                                    current_results_top_n = current_results[current_results[top_quantile_name].isin(top_quantiles[q_i])].groupby(["date", group_feature])[mean_feature].mean()
                                    current_results_bottom_n = current_results[current_results[bottom_quantile_name].isin(bottom_quantiles[q_i])].groupby(["date", group_feature])[mean_feature].mean()
                                    current_group_result_digest[f"top_{quantile}_{mean_feature}_mean"] = current_results_top_n
                                    current_group_result_digest[f"bottom_{quantile}_{mean_feature}_mean"] = current_results_bottom_n
                                    current_group_result_digest[f"top_minus_bottom_{quantile}_{mean_feature}_mean"] = current_results_top_n - current_results_bottom_n
                                    current_group_result_digest[f"mean_{quantile}_{mean_feature}_mean"] = (current_results_top_n + current_results_bottom_n)/2
                                    current_group_result_digest[f"top_{quantile}_n_stocks_chosen"] = current_results[current_results[top_quantile_name].isin(top_quantiles[q_i])].groupby(["date", group_feature])["gvkey"].count()
                                    current_group_result_digest[f"bottom_{quantile}_n_stocks_chosen"] = current_results[current_results[bottom_quantile_name].isin(bottom_quantiles[q_i])].groupby(["date", group_feature])["gvkey"].count()

                        if group_feature != "all":
                            current_results_group_result_n_stocks = current_results.groupby(["date", group_feature])["gvkey"].nunique()

                        else:
                            current_results_group_result_n_stocks = current_results.groupby(["date"])["gvkey"].nunique()
                        current_group_result_digest["n_stocks_in_group"] = current_results_group_result_n_stocks
                        current_group_result_digest.reset_index(inplace=True)

                        if g_i == 0:
                            all_group_result_digest = current_group_result_digest
                            continue
                        all_group_result_digest = pd.concat([all_group_result_digest, current_group_result_digest])

                    all_group_result_digest["min_quantile"] = min_max_quantiles[0]
                    all_group_result_digest["max_quantile"] = min_max_quantiles[1]
                    if n_i == 0:
                        all_quantiles_result_digest = all_group_result_digest
                        continue
                    all_quantiles_result_digest = pd.concat([all_quantiles_result_digest, all_group_result_digest])

                all_quantiles_result_digest["train_file"] = train_file
                if first_results:
                    all_train_file_results_digest = all_quantiles_result_digest
                    first_results = False
                    continue
                all_train_file_results_digest = pd.concat([all_train_file_results_digest, all_quantiles_result_digest])
            else:
                sys.stdout.write(f"Not enough models found for {region} for {train_file} in {year}\n")
                continue
                
        all_train_file_results_digest["split_year"] = year
        if ((y_i == 0) and (region == "US")):
            all_results_digest = all_train_file_results_digest
            continue
        if ((year == 2005) and (region == "Europe")) or ((year == 2005) and (region == "Japan")):
            all_results_digest = all_train_file_results_digest
            continue
        all_results_digest = pd.concat([all_results_digest, all_train_file_results_digest])
        
    model_name_and_region = f"{models}_ensemble_{region}"
        
    if max_trr_5_fwd:
        model_name_and_region += f"max_trr_5_fwd_ar_{max_trr_5_fwd}"
                    
    if min_trr_5_fwd:
        model_name_and_region += f"min_trr_5_fwd_ar_{min_trr_5_fwd}"
        
    if args["no_conviction"]:
        model_name_and_region += "_no_conviction"

    all_results_digest["model"] = model_name_and_region
    all_results_digest["conviction"] = (not args["no_conviction"])
    all_results_digest.to_parquet(f"results/digests/{model_name_and_region}_digest.parquet")