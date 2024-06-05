

def main(start_year, finish_year, drop_years=1, no_fundamentals=False, annual_fundamentals=True):
    import sys
    import pandas as pd
    import numpy as np
    import os

    pd.set_option('display.max_columns', 100)

    years = list(range(start_year, finish_year + 1))
    
    sys.stdout.write(str(years) + "\n")
    
    data_list = []
    for year in years:
        data_list.append(pd.read_parquet(f'data/unprocessed/csrp/data_{year}.parquet', engine='pyarrow'))
    data = pd.concat(data_list, ignore_index=True)

    sys.stdout.write("Loaded data \n")
    sys.stdout.write("Data shape: " + str(data.shape) + "\n")

    csrp_dict = {
        "date" : "date",
        "permco" : "permco",
        "permno" : "permno",
        "issuno" : "issue_number",
        "comnam" : "company_name",
        "cusip" : "cusip",
        "ticker" : "ticker",
        "shrcd" : "share_code",
        "shrcls" : "share_class",
        "exchcd": "exchange_code",
        "primexch": "primary_exchange",
        "prc" : "price_close_usd",
        "vol" : "volume_shares",
        "openprc": "price_open_usd",
        #"ret" : "trr_1",
        "shrout" : "shares_outstanding",
        "divamt" : "dividend_yield_usd",
        "cfacpr" : "adjust_price", #DIVIDE BY THIS
        "cfacshr" : "adjust_shares" #MULTIPLY BY THIS
    }

    data.columns = [x.lower() for x in data.columns]
    data = data[csrp_dict.keys()]
    data.rename(columns=csrp_dict, inplace=True)

    data['date'] = pd.to_datetime(data['date'])
    data["permco"] = data["permco"].astype(str)
    data.dropna(subset=["permco"], inplace=True)
    data = data.sort_values(by=['date', 'permco'])
    data.reset_index(drop=True, inplace=True)

    data = data[data["volume_shares"].notnull()]
    data = data[data["volume_shares"] != 0]
    data = data[data["price_close_usd"].notnull()]
    data = data[data["price_close_usd"] != 0]
    data = data[data["shares_outstanding"].notnull()]
    data = data[data["shares_outstanding"] != 0]

    data = data[data["share_code"].astype(str).str[0] == "1"]
    data = data[data["share_code"].astype(str).str[1].isin(["1", "2", "3"])]
    data = data[data["exchange_code"] >= 0]
    data["price_close_usd"] = abs(data["price_close_usd"])

    data["volume_usd_1"] = data["volume_shares"] * data["price_close_usd"]
    data["market_cap_usd"] = data["price_close_usd"] * data["shares_outstanding"]*1000 #Adjusted for thousands, in CSRP daily data

    data["total_market_cap_usd"] = data.groupby(["permco", "date"])["market_cap_usd"].transform("sum")

    data["total_volume_usd_1"] = data.groupby(["permco", "date"])["volume_usd_1"].transform("sum")

    data = data.sort_values(["date", "permco", "share_class", "permno"]).drop_duplicates(["date", "permco"])

    data = data.sort_values(by=['date', 'permco'])

    data["market_cap_usd"] = data["total_market_cap_usd"]
    data["volume_usd_1"] = data["total_volume_usd_1"]
    data.drop(columns = ["total_market_cap_usd", "total_volume_usd_1"], inplace=True)

    data.reset_index(drop=True, inplace=True)

    # TECHNICAL HORIZONS:
    data["volume_usd_2"]   = data.groupby("permco")["volume_usd_1"].rolling(2  ).mean().reset_index(level=0, drop=True)
    data["volume_usd_3"]   = data.groupby("permco")["volume_usd_1"].rolling(3  ).mean().reset_index(level=0, drop=True)
    data["volume_usd_5"]   = data.groupby("permco")["volume_usd_1"].rolling(5  ).mean().reset_index(level=0, drop=True)
    data["volume_usd_10"]  = data.groupby("permco")["volume_usd_1"].rolling(10 ).mean().reset_index(level=0, drop=True)
    data["volume_usd_20"]  = data.groupby("permco")["volume_usd_1"].rolling(20 ).mean().reset_index(level=0, drop=True)
    data["volume_usd_30"]  = data.groupby("permco")["volume_usd_1"].rolling(30 ).mean().reset_index(level=0, drop=True)
    data["volume_usd_60"]  = data.groupby("permco")["volume_usd_1"].rolling(60 ).mean().reset_index(level=0, drop=True)
    data["volume_usd_90"]  = data.groupby("permco")["volume_usd_1"].rolling(90 ).mean().reset_index(level=0, drop=True)
    data["volume_usd_120"] = data.groupby("permco")["volume_usd_1"].rolling(120).mean().reset_index(level=0, drop=True)
    data["volume_usd_240"] = data.groupby("permco")["volume_usd_1"].rolling(240).mean().reset_index(level=0, drop=True)

    data["adjust_price"] = pd.to_numeric(data["adjust_price"], errors="coerce")
    data["price_close_usd"] = pd.to_numeric(data["price_close_usd"], errors="coerce")
    data.dropna(subset="adjust_price", inplace=True)
    data["price_close_usd_adj"] = data["price_close_usd"] / data["adjust_price"]

    data["price_close_usd_adj"] = pd.to_numeric(data["price_close_usd_adj"], errors="coerce")
    data["market_cap_usd"] = pd.to_numeric(data["market_cap_usd"], errors="coerce")

    data.dropna(subset=["price_close_usd_adj"], inplace=True)
    data["trr_1"] = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=1).reset_index(level=0, drop=True)


    data.loc[data.groupby("permco").head(1).index, 'trr_1'] = 0

    data.reset_index(drop=True, inplace=True)

    # ALL CALCULATED WITH PCT_CHANGE
    data["trr_2"]   = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=2  ).reset_index(level=0, drop=True)
    data["trr_3"]   = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=3  ).reset_index(level=0, drop=True)
    data["trr_5"]   = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=5  ).reset_index(level=0, drop=True)
    data["trr_10"]  = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=10 ).reset_index(level=0, drop=True)
    data["trr_20"]  = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=20 ).reset_index(level=0, drop=True)
    data["trr_30"]  = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=30 ).reset_index(level=0, drop=True)
    data["trr_60"]  = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=60 ).reset_index(level=0, drop=True)
    data["trr_90"]  = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=90 ).reset_index(level=0, drop=True)
    data["trr_120"] = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=120).reset_index(level=0, drop=True)
    data["trr_240"] = data.groupby(["permco"])["price_close_usd_adj"].pct_change(periods=240).reset_index(level=0, drop=True)

    data["trr_1"] = np.log(data["trr_1"] + 1)
    data["trr_2"] = np.log(data["trr_2"] + 1)
    data["trr_3"] = np.log(data["trr_3"] + 1)
    data["trr_5"] = np.log(data["trr_5"] + 1)
    data["trr_10"] = np.log(data["trr_10"] + 1)
    data["trr_20"] = np.log(data["trr_20"] + 1)
    data["trr_30"] = np.log(data["trr_30"] + 1)
    data["trr_60"] = np.log(data["trr_60"] + 1)
    data["trr_90"] = np.log(data["trr_90"] + 1)
    data["trr_120"] = np.log(data["trr_120"] + 1)
    data["trr_240"] = np.log(data["trr_240"] + 1)

    data["volatility_5"]   = data.groupby("permco")["trr_1"].rolling(5  ).std().reset_index(level=0, drop=True)
    data["volatility_10"]  = data.groupby("permco")["trr_1"].rolling(10 ).std().reset_index(level=0, drop=True)
    data["volatility_20"]  = data.groupby("permco")["trr_1"].rolling(20 ).std().reset_index(level=0, drop=True)
    data["volatility_30"]  = data.groupby("permco")["trr_1"].rolling(30 ).std().reset_index(level=0, drop=True)
    data["volatility_60"]  = data.groupby("permco")["trr_1"].rolling(60 ).std().reset_index(level=0, drop=True)
    data["volatility_90"]  = data.groupby("permco")["trr_1"].rolling(90 ).std().reset_index(level=0, drop=True)
    data["volatility_120"] = data.groupby("permco")["trr_1"].rolling(120).std().reset_index(level=0, drop=True)
    data["volatility_240"] = data.groupby("permco")["trr_1"].rolling(240).std().reset_index(level=0, drop=True)

    sys.stdout.write("Added horizons\n")
    sys.stdout.write("Data shape: " + str(data.shape) + "\n")

    if no_fundamentals:
        sys.stdout.write("Not loading fundamentals \n")
        sys.stdout.write("Writing data \n")
    
        for index, year in enumerate(years):
            if index < drop_years:
                sys.stdout.write("Not writing data", year, "\n")
                continue
            sys.stdout.write("Writing data", year, "\n")
            data[data["date"].dt.year == year].to_parquet(f'data/processed/csrp/us_data_{year}_processed_n_pct_mc.parquet', index=False)

        sys.stdout.write("Finished processing \n")
        return

    # LOAD PROCESSED FUNDAMENTALS DATA:
    if annual_fundamentals:
        fundamentals_data = pd.read_parquet('data/processed/csrp/csrp_compustat_fundamentals_processed_annual.parquet')
    else:
        fundamentals_data = pd.read_parquet('data/processed/csrp/csrp_compustat_fundamentals_processed_2.parquet')

    fundamentals_data.drop(columns="company_name", inplace=True, errors="ignore")

    # CALCULATE FUNDAMENTAL FEATURES AND RATIOS:    
    fundamentals_data['date'] = pd.to_datetime(fundamentals_data['date'])
    fundamentals_data["permco"] = fundamentals_data["permco"].astype(str)
    fundamentals_data = fundamentals_data.sort_values(by=['date', 'permco'])

    fundamentals_data = fundamentals_data[fundamentals_data['date'] >= pd.Timestamp("1950-06-01")]
    data = data[data["date"] >= pd.Timestamp("1950-06-01")]

    # MERGE FUNDAMENTALS AND GLOBAL DATA:
    data.reset_index(drop=True, inplace=True)
    fundamentals_data.reset_index(drop=True, inplace=True)
    fundamentals_data.sort_values(by=['date', 'permco'], inplace=True)
    data.sort_values(by=['date', 'permco'], inplace=True)

    joined_data = pd.merge_asof(data, fundamentals_data, on='date', by='permco', allow_exact_matches=False)

    joined_data["linkdt"] = pd.to_datetime(joined_data["linkdt"])
    joined_data["linkenddt"] = pd.to_datetime(joined_data["linkenddt"], errors='coerce')

    joined_data = joined_data[((joined_data["date"] >= joined_data["linkdt"]) & (joined_data["date"] <= joined_data["linkenddt"])) |
                         ((joined_data["date"] >= joined_data["linkdt"]) & (joined_data["linkenddt"].isna()))]

    joined_data["earnings_to_price"] = joined_data["net_income"]*1000000 / joined_data["market_cap_usd"]
    joined_data["earnings_bex_to_price"] = joined_data["income_bex"]*1000000 / joined_data["market_cap_usd"]
    joined_data["sales_to_price"] = joined_data["net_sales"]*1000000 / joined_data["market_cap_usd"]
    joined_data["book_to_price"] =  joined_data["stockholders_equity"]*1000000 / joined_data["market_cap_usd"]

    joined_data["dividend_yield"] = joined_data["dividend_yield_usd"] / joined_data["price_close_usd"]
    joined_data["dividend_yield"] = joined_data.sort_values(by="date", ascending=True).groupby(['permco'])[['dividend_yield', 'date']].rolling(f"{30*11}D", on='date').sum().reset_index().set_index('level_1')['dividend_yield']

    sys.stdout.write("After fundamentals\n")
    sys.stdout.write("Data shape: " + str(joined_data.shape) + "\n")
    
    # KEEP ONLY FINAL FEATURES:
    final_features_df = pd.read_csv('final_features_csrp_n.csv', delimiter=';')
    final_features = final_features_df[final_features_df['final_feature'].notnull()]["final_feature"].to_list()

    joined_data["permno"] = joined_data["permno_x"]
    joined_data.drop(columns=["permno_x"], inplace=True)

    sys.stdout.write("Not in joined_data:\n")
    sys.stdout.write(str(set(final_features) - set(joined_data.columns.tolist()) + "\n"))
    final_features = [feature for feature in final_features if feature in joined_data.columns.tolist()]
    final_features_df = final_features_df[final_features_df['final_feature'].isin(final_features)]

    joined_data = joined_data[final_features]

    # CONVERT DATA TYPES, ALSO DONE ON IMPORT:
    dtype_dict = dict(zip(final_features_df["final_feature"], final_features_df["feature_type"]))
    del dtype_dict['date']

    for index, feature in final_features_df.iterrows():
        if pd.isnull(feature["final_feature"]) or feature["final_feature"] == "date":
            continue
        if feature["feature_type"] == "string":
            joined_data[feature["final_feature"]] = joined_data[feature["final_feature"]].astype(str)
        elif feature["feature_type"] == "int":
            joined_data[feature["final_feature"]] = joined_data[feature["final_feature"]].astype(pd.Int64Dtype())
        elif feature["feature_type"] == "float":
            joined_data[feature["final_feature"]] = joined_data[feature["final_feature"]].astype(float)
        if feature["categorical"] == "True":
            joined_data[feature["final_feature"]] = joined_data[feature["final_feature"]].astype("category")

    sys.stdout.write("End\n")
    sys.stdout.write("Data shape: " + str(joined_data.shape) + "\n")

    sys.stdout.write("Writing data \n")
    
    for index, year in enumerate(years):
        if index < drop_years:
            sys.stdout.write("Not writing data", year, "\n")
            continue
        sys.stdout.write("Writing data", year, "\n")
        if annual_fundamentals:
            joined_data[joined_data["date"].dt.year == year].to_parquet(f'data/processed/csrp/us_data_{year}_annual_fund_processed_n_pct.parquet', index=False)
        else:
            joined_data[joined_data["date"].dt.year == year].to_parquet(f'data/processed/csrp/us_data_{year}_processed_n_pct.parquet', index=False)

    sys.stdout.write("Finished processing \n")