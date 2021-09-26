import concurrent.futures
import datetime as dt
import os
import time
from functools import partial
from os import listdir
from os.path import isfile, join

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from pandas_datareader import data as web

# save stocks to csv


def save_to_csv_from_yahoo(folder, ticker, syear, smonth, sday, eyear, emonth, eday):
    start = dt.datetime(syear, smonth, sday)
    end = dt.datetime(eyear, emonth, eday)

    try:
        print("Get Data for:", ticker)
        df = web.DataReader(ticker, 'yahoo', start, end)['Adj Close']
        time.sleep(10)
        df.to_csv(folder + ticker + '.csv')
        return True
    except Exception as ex:
        # stocks_not_downloaded.append(ticker)
        print("Couldn't get Data for:", ticker)
        return False


# get dateframe from .csv


def get_stock_df_from_csv(folder, ticker):
    try:
        df = pd.read_csv(folder + ticker + '.csv')
    except FileNotFoundError:
        print("File for {} doesn't exist!".format(ticker))
    else:
        return df


# return named column data from .csv


def get_column_from_csv(file, col_name):
    try:
        df = pd.read_csv(file)
    except FileNotFoundError:
        print("File doesn't exist!")
    else:
        return df[col_name]


# parallel save stock to csv


def parallel_save_to_csv_from_yahoo(folder, tickers):
    failures = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_save_csv = {executor.submit(
            save_to_csv_from_yahoo, folder, ticker, 2017, 1, 1, 2021, 9, 23): ticker for ticker in tickers}

        for future in concurrent.futures.as_completed(future_to_save_csv):
            ticker = future_to_save_csv[future]
            result = future.result()

            if result is False:
                failures.append(ticker)
    if len(failures) > 0:
        print("Failed to download:")
        print(failures)


tickers = get_column_from_csv(
    "C:\\DEV\\Python\\Stock_Sandbox\\stock_list\\Wilshire-5000-Stocks.csv", 'Ticker')

folder = "C:\\DEV\\Python\\Stock_Sandbox\\stock_data\\"

print("Downloading Stock Data")

# failed = ['ACHN', 'BDGE', 'BMCH', 'ANH', 'AKER', 'APEX', 'AHC', 'AEGN', 'ALXN', 'AIMT', 'AAXN', 'ACIA', 'CGIX', 'BEAT', 'ARA', 'AKCA', 'AIII', 'BSTC', 'CBMG', 'CLGX', 'BPFH', 'CLNC', 'BASI', 'CLNY', 'CPST', 'CKH', 'DRAD', 'CLCT', 'CXO', 'EGOV', 'CMD', 'DNKN', 'EIGI', 'CATM', 'EIDX', 'CUB', 'FBM', 'CTB', 'FFG', 'EV', 'CTRA', 'CBSA', 'GV', 'GRIF', 'FBSS', 'GFN', 'GEN', 'FLIR', 'GLUU', 'GEC', 'ETM', 'GLIBA', 'FSCT', 'FRAN', 'GNMK', 'HMSY', 'FPRX', 'IPHI', 'HPR', 'IRET', 'GTT', 'FIT', 'HWCC', 'HNR', 'HDS', 'MGEN', 'ISNS', 'HCFT', 'KNL', 'LEAF', 'MIK', 'KOOL', 'NEOS', 'NK', 'NTN', 'LMNX', 'PEIX', 'MSGN', 'NAV', 'PLT', 'PICO', 'OTEL', 'MDLY', 'MLND', 'MTSC', 'PE', 'RESI', 'PTVCB', 'NGHC', 'RNET', 'NVUS', 'PTI', 'PRSC', 'PS', 'PTVCA', 'PIH', 'PDLI', 'RP', 'PRGX', 'PRAH', 'RLH', 'STAY', 'PRSP', 'QEP', 'SNSS', 'TAT', 'SONA', 'SVMK', 'STND', 'PRCP', 'TCO', 'TNAV', 'SYX', 'SYNC', 'TLF', 'VRTU', 'TTS', 'VAR', 'WDR', 'ROKA', 'WIFI', 'TPRE', 'TCF', 'TRCH', 'YRCW', 'TPCO', 'ZAGG', 'TIF', 'WPX', 'USAT', 'TECD', 'TRXC', 'WYND', 'UROV', 'XAN', 'WRTC']

# parallel_save_to_csv_from_yahoo(folder, failed)

print("Finished")
