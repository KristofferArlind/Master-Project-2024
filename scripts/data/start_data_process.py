import argparse
import sys
import os

import us_data_n_pct
import europe_data_n_pct
import japan_data_n_pct

parser = argparse.ArgumentParser()

parser.add_argument("-region", help="Process region (EU, US, Japan)", default="US")
parser.add_argument("-start", help="Year to start processing on", type=int, default=2020)
parser.add_argument("-last", help="Year to finish processing on", type=int, default=2024)
parser.add_argument("-max_years", help="Max years to process at a time", type=int, default=4)
parser.add_argument("-overlap", help="Years of overlap between intervals", type=int, default=1)
parser.add_argument("--no_fundamentals", help="Do not join fundamental data on the daily data", action="store_true")
parser.add_argument("--annual_fundamentals", help="Use annual fundamentals rather than quarterly", action="store_true")


args = parser.parse_args()
args = vars(args)

overlap = int(args["overlap"])
max_years = int(args["max_years"])
last_year = int(args["last"])
first_year = int(args["start"]) 
no_fundamentals = args["no_fundamentals"]
annual_fundamentals = args["annual_fundamentals"]

os.chdir("../")
sys.stdout.write("Current working directory: " + os.getcwd() + "\n")


current_last_year = last_year
current_first_year = max(current_last_year - max_years + 1, first_year)
   
if args["region"] == "US":
    while True:  
        sys.stdout.write(f"Processing US CSRP data from {current_first_year} to {current_last_year}")
        if current_first_year <= first_year:
            us_data_n_pct.main(current_first_year, current_last_year, drop_years=0, no_fundamentals=no_fundamentals, annual_fundamentals=annual_fundamentals)
            break
        us_data_n_pct.main(current_first_year, current_last_year, drop_years=overlap, no_fundamentals=no_fundamentals, annual_fundamentals=annual_fundamentals)
        if current_first_year <= first_year:
            break
        current_last_year = current_first_year - 1 + overlap
        current_first_year = max(current_last_year - max_years + 1, first_year)

if args["region"] == "EU":
    while True:  
        sys.stdout.write(f"Processing EU data from {current_first_year} to {current_last_year}")
        if current_first_year <= first_year:
            europe_data_n_pct.main(current_first_year, current_last_year, drop_years=0, annual_fundamentals=annual_fundamentals)
            break
        europe_data_n_pct.main(current_first_year, current_last_year, drop_years=overlap, annual_fundamentals=annual_fundamentals)
        if current_first_year <= first_year:
            break
        current_last_year = current_first_year - 1 + overlap
        current_first_year = max(current_last_year - max_years + 1, first_year)
        
if args["region"] == "Japan":
    while True:  
        sys.stdout.write(f"Processing Japan data from {current_first_year} to {current_last_year}")
        if current_first_year <= first_year:
            japan_data_n_pct.main(current_first_year, current_last_year, drop_years=0, annual_fundamentals=annual_fundamentals)
            break
        japan_data_n_pct.main(current_first_year, current_last_year, drop_years=overlap, annual_fundamentals=annual_fundamentals)
        if current_first_year <= first_year:
            break
        current_last_year = current_first_year - 1 + overlap
        current_first_year = max(current_last_year - max_years + 1, first_year)

else:
    sys.stdout.write("Invalid region")
    exit(1)