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

PATH = "C:\\DEV\\Python\\Stock_Sandbox\\stock_data\\"
S_YEAR = 2017
S_MONTH = 1
S_DAY = 3
S_DATE_STR = f"{S_YEAR}-{S_MONTH}-{S_DAY}"
S_DATE_DATETIME = dt.datetime(S_YEAR, S_MONTH, S_DAY)

E_YEAR = 2021
E_MONTH = 9
E_DAY = 23
E_DATE_STR = f"{E_YEAR}-{E_MONTH}-{E_DAY}"
E_DATE_DATETIME = dt.datetime(E_YEAR, E_MONTH, E_DAY)

files = [x for x in listdir(PATH) if isfile(join(PATH, x))]
tickers = [os.path.splitext(x)[0] for x in files]

def get_df_from_csv(ticker):
    try:
        df = pd.read_csv(PATH + ticker + '.csv')
    except FileNotFoundError:
        print(f"File: {ticker}.csv does not exist!")
    else:
        return df

def save_df_to_csv(df, ticker):
    df.to_csv(PATH + ticker + '.csv')


def delete_unnamed_columns(df):
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    return df

def add_daily_return_to_df(df, ticker):
    # (close / previous_close) - 1
    df['daily_return'] = (df['Adj Close'] / df['Adj Close'].shift(1)) - 1
    # df.to_csv(PATH + ticker + '.csv')
    return df

def get_roi_defined_time(df):
    # Investing 100 and have 200 in 5 years
    # ROI = End value (200) - Initial (100) / Initial = 1
    # ROI = 1 * Initial = 100
    df['Date'] = pd.to_datetime(df['Date'])
    start_val = df[df['Date'] == S_DATE_STR]['Adj Close'][0]
    end_val = df[df['Date'] == E_DATE_STR]['Adj Close'][0]
    print("Initial Price:", start_val)
    print("Final Price:", end_val)

    roi = (end_val - start_val) / start_val
    return roi

def get_cov(stock_df):
    mean = stock_df['Adj Close'].mean()
    sd = stock_df['Adj Close'].std()

    cov = sd / mean

    return cov

# stock_df = pd.DataFrame(tickers, columns=['Tickers'])

# print(stock_df)

# print(tickers[0])

# stock_a = get_df_from_csv(tickers[0])

# print(stock_a)

# add_daily_return_to_df(stock_a, tickers[0])

# print(stock_a)


# stock_a = delete_unnamed_columns(stock_a)

# save_df_to_csv(stock_a, tickers[0])

for ticker in tickers:
    print("Working on: ", ticker)
    stock_df = get_df_from_csv(ticker)
    stock_df = add_daily_return_to_df(stock_df, ticker)
    stock_df = delete_unnamed_columns(stock_df)
    save_df_to_csv(stock_df, ticker)
