

def main(start_year, finish_year, drop_years=1, annual_fundamentals=True):
    import sys
    import pandas as pd
    import numpy as np
    import os

    pd.set_option('display.max_columns', 100)

    years = list(range(start_year, finish_year + 1))
    
    sys.stdout.write(str(years) + "\n")
    
    data_list = []
    for year in years:
        data_list.append(pd.read_parquet(f'data/unprocessed/global_data_{year}.parquet'))
    global_data = pd.concat(data_list, ignore_index=True)

    sys.stdout.write("Loaded data \n")
    sys.stdout.write("Data shape: " + str(global_data.shape) + "\n")

    column_name_dict = {'gvkey': 'gvkey',
                        'iid': 'iid',
                        'datadate': 'date',
                        'conm': 'company_name',
                        'epf': 'earnings_participation_flag',
                        'curcddv': 'currency_dividend',
                        'cheqv': 'cash_equivalent',
                        'div': 'dividend_loc',
                        'divd': 'dividend_cash',
                        'divsp': 'dividend_special',
                        'anncdate': 'dividend_declare_date',
                        'cheqvpaydate': 'cash_equivalent_pay_date',
                        'divsppaydate': 'dividend_special_pay_date',
                        'recorddate': 'dividend_record_date',
                        'curcdd': 'currency',
                        'ajexdi': 'adjustment_factor',
                        'cshoc': 'shares_outstanding',
                        'cshtrd': 'volume_shares',
                        'prccd': 'price_close',
                        'prchd': 'price_high',
                        'prcld': 'price_low',
                        'prcstd': 'price_status_code',
                        'trfd': 'total_return_factor',
                        'exchg': 'exchange_code',
                        'secstat': 'security_status',
                        'tpci': 'issue_type_code',
                        'cik': 'cik',
                        'fic': 'foreign_incorporation_code',
                        'add1': 'address_line_1',
                        'add2': 'address_line_2',
                        'add3': 'address_line_3',
                        'add4': 'address_line_4',
                        'addzip': 'address_zip',
                        'busdesc': 'business_description',
                        'city': 'city',
                        'conml': 'company_legal_name',
                        'costat': 'company_status',
                        'county': 'county',
                        'dlrsn': 'deletion_reason',
                        'ein': 'ein',
                        'fax': 'fax',
                        'fyrc': 'fiscal_year_end_month',
                        'ggroup': 'ggroup',
                        'gind': 'gind',
                        'gsector': 'gsector',
                        'gsubind': 'gsubind',
                        'idbflag': 'idb_flag',
                        'incorp': 'incorporation_code',
                        'loc': 'country_hq',
                        'naics': 'naics',
                        'phone': 'phone',
                        'prican': 'primary_canada',
                        'prirow': 'primary_row',
                        'priusa': 'primary_usa',
                        'sic': 'sic',
                        'spcindcd': 'sp_industry_code',
                        'spcseccd': 'sp_sector_code',
                        'spcsrc': 'sp_quality',
                        'state': 'state',
                        'stko': 'stko',
                        'weburl': 'web_url',
                        'dldte': 'download_date',
                        'ipodate': 'ipo_date'}


    global_data = global_data[column_name_dict.keys()]
    global_data.rename(columns=column_name_dict, inplace=True)

    global_data = global_data.sort_values(by=['date', 'gvkey', 'iid'])

    global_data = global_data[global_data["volume_shares"].notnull()]
    global_data = global_data[global_data["volume_shares"] != 0]
    global_data = global_data[global_data["price_close"].notnull()]
    global_data = global_data[global_data["price_close"] != 0]

    # CURRENCY DATA:
    forex_data = pd.read_parquet('data/forex/forex_data_daily.parquet', engine='pyarrow')
    forex_data["date"] = pd.to_datetime(forex_data["date"])
    forex_data["currency"] = forex_data["currency"].astype("category")
    forex_data = forex_data.sort_values("date")
    forex_dict = forex_data.set_index(['date', 'currency'])['to_usd'].to_dict()

    forex_data_12m = pd.read_parquet('data/forex/forex_data_12m.parquet', engine='pyarrow')
    forex_data_12m["date"] = pd.to_datetime(forex_data_12m["date"])
    forex_data_12m["currency"] = forex_data_12m["currency"].astype("category")
    forex_data_12m = forex_data_12m.sort_values("date")
    forex_12m_dict = forex_data_12m.set_index(['date', 'currency'])['to_usd_12m'].to_dict()

    global_data = global_data[~global_data['currency'].isnull()]

    global_data['date'] = pd.to_datetime(global_data['date'])
    global_data.reset_index(drop=True, inplace=True)

    global_data = global_data[global_data['currency'].isin(forex_data["currency"].unique())]
    global_data = global_data[global_data['currency_dividend'].isin(forex_data["currency"].unique()) | global_data['currency_dividend'].isnull()]

    data = global_data

    # REMOVE OTC EXCHANGES AND NON NORMAL SHARES:
    disallowed_exchange_codes = [13, 19, 229, 290]
    data = data[~data["exchange_code"].isin(disallowed_exchange_codes)]

    data["issue_type_code"] = data["issue_type_code"].astype(str)
    data = data[data["issue_type_code"].isin(["0", "1", "4", "Q", "8"])]
    data = data[data["earnings_participation_flag"] == "Y"]
    data = data[data["security_status"] == "A"]

    # ADJUST PRICE_CLOSE:
    data["price_close_adj"] = data["price_close"] / data["adjustment_factor"]

    def convert_to_usd(group):
        date, currency = group.name
        if (date, currency) not in forex_dict:
            tries = 0
            while (date, currency) not in forex_dict:
                date = date - pd.Timedelta(1, "d")
                tries += 1
                if tries > 10:
                    group.loc[:] = np.nan
                    return group
        to_usd_rate = forex_dict[(date, currency)]
        group = group * to_usd_rate
        return group
    
    def convert_to_usd_12m(group):
        date, currency = group.name
        if (date, currency) not in forex_12m_dict:
            tries = 0
            while (date, currency) not in forex_12m_dict:
                date = date - pd.Timedelta(1, "d")
                tries += 1
                if tries > 10:
                    group.loc[:] = np.nan
                    return group
        to_usd_rate = forex_12m_dict[(date, currency)]
        group = group * to_usd_rate
        return group

    # CONVERT TO USD:
    data["price_close_usd"] = data.groupby(["date", "currency"])["price_close"].transform(convert_to_usd)

    data["market_cap_usd"] = data["price_close_usd"] * data["shares_outstanding"]
    data["volume_usd_1"] = data["volume_shares"] * data["price_close_usd"]

    data["gvkey"] = data["gvkey"].astype(str)
    data["iid"] = data["iid"].astype(str)


    # REMOVE DUPLICATE LISTINGS. KEEP HIGHEST AVG DAILY VOLUME, SUM VOLUMES:
    data["total_volume_usd_1"] = data.groupby(["gvkey", "date"])["volume_usd_1"].transform("sum")
    data["total_market_cap_usd"] = data.groupby(["gvkey", "date"])["market_cap_usd"].transform("sum")

    # KEEP ONLY JPY
    data = data[data["currency"].isin(["JPY"])]

    data = data[(data["iid"] == data["primary_row"]) | (data["primary_row"].isna())]

    avg_volumes = data.groupby(['gvkey', 'iid']).apply(lambda x: x['volume_usd_1'].mean()).reset_index()
    avg_volumes.columns = ['gvkey', 'iid', 'avg_volume']
    max_volumes = avg_volumes.groupby('gvkey').apply(lambda x: x[x['avg_volume'] == x['avg_volume'].max()]).reset_index(drop=True)

    data = pd.merge(data, max_volumes[['gvkey', 'iid']], on=['gvkey', 'iid'], how='inner')
    data.drop_duplicates(subset=["date", "gvkey"], inplace=True)

    data["volume_usd_1"] = data["total_volume_usd_1"]
    data.drop(labels=["total_volume_usd_1"], axis="columns", inplace=True)

    data["market_cap_usd"] = data["total_market_cap_usd"]
    data.drop(labels=["total_market_cap_usd"], axis="columns", inplace=True)

    print((data.groupby('gvkey')['iid'].nunique() > 1).sum())

    sys.stdout.write("Removed duplicate listings\n")
    sys.stdout.write("Data shape: " + str(data.shape) + "\n")

    # TECHNICAL HORIZONS:
    data["volume_usd_2"]   = data.groupby("gvkey")["volume_usd_1"].rolling(2  ).mean().reset_index(level=0, drop=True)
    data["volume_usd_3"]   = data.groupby("gvkey")["volume_usd_1"].rolling(3  ).mean().reset_index(level=0, drop=True)
    data["volume_usd_5"]   = data.groupby("gvkey")["volume_usd_1"].rolling(5  ).mean().reset_index(level=0, drop=True)
    data["volume_usd_10"]  = data.groupby("gvkey")["volume_usd_1"].rolling(10 ).mean().reset_index(level=0, drop=True)
    data["volume_usd_20"]  = data.groupby("gvkey")["volume_usd_1"].rolling(20 ).mean().reset_index(level=0, drop=True)
    data["volume_usd_30"]  = data.groupby("gvkey")["volume_usd_1"].rolling(30 ).mean().reset_index(level=0, drop=True)
    data["volume_usd_60"]  = data.groupby("gvkey")["volume_usd_1"].rolling(60 ).mean().reset_index(level=0, drop=True)
    data["volume_usd_90"]  = data.groupby("gvkey")["volume_usd_1"].rolling(90 ).mean().reset_index(level=0, drop=True)
    data["volume_usd_120"] = data.groupby("gvkey")["volume_usd_1"].rolling(120).mean().reset_index(level=0, drop=True)
    data["volume_usd_240"] = data.groupby("gvkey")["volume_usd_1"].rolling(240).mean().reset_index(level=0, drop=True)

    # ADJUSTED RETURNS
    data["total_return_factor"] = data["total_return_factor"].fillna(1)
    data["price_adj_return_factor"] = data["price_close_adj"] * data["total_return_factor"]


    data.reset_index(drop=True, inplace=True)
    data.dropna(subset=["price_adj_return_factor"], inplace=True)
    data["trr_1"] = data.groupby(["gvkey"])["price_adj_return_factor"].pct_change(1).reset_index(level=0, drop=True)

    data.loc[data.groupby("gvkey").head(1).index, 'trr_1'] = 0
    data.reset_index(drop=True, inplace=True)


    # ALL CALCULATED WITH PCT_CHANGE
    data["trr_2"]   = data.groupby("gvkey")["price_adj_return_factor"].pct_change(periods=2  ).reset_index(level=0, drop=True)
    data["trr_3"]   = data.groupby("gvkey")["price_adj_return_factor"].pct_change(periods=3  ).reset_index(level=0, drop=True)
    data["trr_5"]   = data.groupby("gvkey")["price_adj_return_factor"].pct_change(periods=5  ).reset_index(level=0, drop=True)
    data["trr_10"]  = data.groupby("gvkey")["price_adj_return_factor"].pct_change(periods=10 ).reset_index(level=0, drop=True)
    data["trr_20"]  = data.groupby("gvkey")["price_adj_return_factor"].pct_change(periods=20 ).reset_index(level=0, drop=True)
    data["trr_30"]  = data.groupby("gvkey")["price_adj_return_factor"].pct_change(periods=30 ).reset_index(level=0, drop=True)
    data["trr_60"]  = data.groupby("gvkey")["price_adj_return_factor"].pct_change(periods=60 ).reset_index(level=0, drop=True)
    data["trr_90"]  = data.groupby("gvkey")["price_adj_return_factor"].pct_change(periods=90 ).reset_index(level=0, drop=True)
    data["trr_120"] = data.groupby("gvkey")["price_adj_return_factor"].pct_change(periods=120).reset_index(level=0, drop=True)
    data["trr_240"] = data.groupby("gvkey")["price_adj_return_factor"].pct_change(periods=240).reset_index(level=0, drop=True)

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

    data["volatility_5"]   = data.groupby("gvkey")["trr_1"].rolling(5  ).std().reset_index(level=0, drop=True)
    data["volatility_10"]  = data.groupby("gvkey")["trr_1"].rolling(10 ).std().reset_index(level=0, drop=True)
    data["volatility_20"]  = data.groupby("gvkey")["trr_1"].rolling(20 ).std().reset_index(level=0, drop=True)
    data["volatility_30"]  = data.groupby("gvkey")["trr_1"].rolling(30 ).std().reset_index(level=0, drop=True)
    data["volatility_60"]  = data.groupby("gvkey")["trr_1"].rolling(60 ).std().reset_index(level=0, drop=True)
    data["volatility_90"]  = data.groupby("gvkey")["trr_1"].rolling(90 ).std().reset_index(level=0, drop=True)
    data["volatility_120"] = data.groupby("gvkey")["trr_1"].rolling(120).std().reset_index(level=0, drop=True)
    data["volatility_240"] = data.groupby("gvkey")["trr_1"].rolling(240).std().reset_index(level=0, drop=True)

    sys.stdout.write("Added horizons\n")
    sys.stdout.write("Data shape: " + str(data.shape) + "\n")

    # LOAD PROCESSED FUNDAMENTALS DATA:
    if annual_fundamentals:
        global_fundamentals_data = pd.read_parquet('data/processed/global_fundamentals_annual_processed.parquet')
    else:
        global_fundamentals_data = pd.read_parquet('data/processed/global_fundamentals_processed.parquet')


    # CALCULATE FUNDAMENTAL FEATURES AND RATIOS:
    fundamentals_data = global_fundamentals_data
    fundamentals_data.drop(columns="company_name", inplace=True, errors="ignore")

    data['date'] = pd.to_datetime(data['date'])
    fundamentals_data['date'] = pd.to_datetime(fundamentals_data['date'])
    fundamentals_data["gvkey"] = fundamentals_data["gvkey"].astype(str)

    # MERGE FUNDAMENTALS AND DAILY DATA:
    data.reset_index(drop=True, inplace=True)
    fundamentals_data.reset_index(drop=True, inplace=True)
    fundamentals_data.sort_values(by=['date', 'gvkey'], inplace=True)
    data.sort_values(by=['date', 'gvkey'], inplace=True)

    joined_data = pd.merge_asof(data, fundamentals_data, on='date', by='gvkey', allow_exact_matches=False)
    
    joined_data["earnings_to_price"] = joined_data["net_income"]*1000000 / joined_data["market_cap_usd"]
    joined_data["earnings_bex_to_price"] = joined_data["income_bex"]*1000000 / joined_data["market_cap_usd"]
    joined_data["sales_to_price"] = joined_data["net_sales"]*1000000 / joined_data["market_cap_usd"]
    joined_data["book_to_price"] =  joined_data["stockholders_equity"]*1000000 / joined_data["market_cap_usd"]


    joined_data["dividend_yield"] = joined_data["dividend_loc"] / joined_data["price_close_usd"]
    joined_data["dividend_yield"] = joined_data.sort_values(by="date").groupby(['gvkey'])[['dividend_yield', 'date']].rolling(f"{30*11}D", on='date').sum().reset_index().set_index('level_1')['dividend_yield']
    joined_data["dividend_yield"] = joined_data.groupby(["date", "currency"])["dividend_yield"].transform(convert_to_usd_12m)

    sys.stdout.write("After fundamentals\n")
    sys.stdout.write("Data shape: " + str(joined_data.shape) + "\n")

    # KEEP ONLY FINAL FEATURES:
    final_features_df = pd.read_csv('final_features_n.csv', delimiter=';')
    final_features = final_features_df[final_features_df['final_feature'].notnull()]["final_feature"].to_list()

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
            joined_data[joined_data["date"].dt.year == year].to_parquet(f'data/processed/japan/japan_data_{year}_annual_fund_processed_n_pct.parquet', index=False)
        else:
            joined_data[joined_data["date"].dt.year == year].to_parquet(f'data/processed/japan/japan_data_{year}_processed_n_pct.parquet', index=False)

    sys.stdout.write("Finished processing \n")