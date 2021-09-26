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


def get_valid_dates(df, sdate, edate):
    try:
        mask = (df['Date'] > sdate) & (df['Date'] <= edate)
        sm_df = df.loc[mask]
        sm_df = sm_df.set_index(['Date'])
        sm_date = sm_df.index.min()
        last_date = sm_df.index.max()

        date_leading = '-'.join(('0' if len(x) < 2 else '') +
                                x for x in sm_date.split('-'))
        date_ending = '-'.join(('0' if len(x) < 2 else '') +
                               x for x in last_date.split('-'))
    except Exception:
        print("Date is corrupt!")
    else:
        return date_leading, date_ending


def get_roi_between_dates(df, sdate, edate):
    try:
        start_val = df.loc[sdate, 'Adj Close']
        end_val = df.loc[edate, 'Adj Close']
        roi = ((end_val - start_val) / start_val)
    except Exception:
        print("Data is corrupt!")
    else:
        return roi


def get_mean_between_dates(df, sdate, edate):
    mask = (df['Date'] > sdate) & (df['Date'] <= edate)
    return df.loc[mask]['Adj Close'].mean()


def get_sd_between_dates(df, sdate, edate):
    mask = (df['Date'] > sdate) & (df['Date'] <= edate)
    return df.loc[mask]['Adj Close'].std()


def get_cov_between_dates(df, sdate, edate):
    mean = get_mean_between_dates(df, sdate, edate)
    sd = get_sd_between_dates(df, sdate, edate)
    return sd / mean


def get_cov_ror(tickers, sdate, edate):
    col_names = ['Ticker', 'COV', 'ROI']
    df = pd.DataFrame(columns=col_names)

    failures = []
    for ticker in tickers:
        print("Working on:", ticker)
        s_df = get_df_from_csv(ticker)
        try:
            sdate2, edate2 = get_valid_dates(s_df, sdate, edate)
            cov = get_cov_between_dates(s_df, sdate2, edate2)
            s_df = s_df.set_index(['Date'])
            roi = get_roi_between_dates(s_df, sdate2, edate2)
            df.loc[len(df.index)] = [ticker, cov, roi]
        except TypeError:
            failures.append(ticker)

    if len(failures) > 0:
        print("Stocks not in date range", failures)

    return df


def merge_df_by_column_name(col_name, sdate, edate, *tickers):
    mult_df = pd.DataFrame()
    for ticker in tickers:
        df = get_df_from_csv(ticker)
        df['Date'] = pd.to_datetime(df['Date'])
        mask = (df['Date'] >= sdate) & (df['Date'] <= edate)

        mult_df[ticker] = df.loc[mask][col_name]

    return mult_df


stock_a = get_df_from_csv(tickers[0])
print(stock_a)

sdate, edate = get_valid_dates(stock_a, '2020-01-01', '2020-12-31')

adj_close_mean = get_mean_between_dates(stock_a, '2020-01-01', '2020-12-31')
sd = get_sd_between_dates(stock_a, '2020-01-01', '2020-12-31')
cov = get_cov_between_dates(stock_a, '2020-01-01', '2020-12-31')

print("Mean:", adj_close_mean)
print("SD:", sd)
print("COV:", cov)

# use proper market dates
# stock_a = stock_a.set_index(['Date'])
# print("Return on Investment:", get_roi_between_dates(stock_a, sdate, edate))

# market_df = get_cov_ror(tickers, '2019-01-01', '2019-12-31')

# print("Top 20 stocks based on ROI")
# top_20_roi = market_df.sort_values(by=['ROI'], ascending=False).head(20)

# print(top_20_roi)

# Create a Correlation Matrix using FAANGS

faang_list = ["FB", "AMZN", "AAPL", "NFLX", "GOOG"]
mult_df = merge_df_by_column_name(
    'daily_return',  '2020-1-1', '2020-12-31', *faang_list)
print(mult_df)

# Generate a Correlation Matrix
print(mult_df.corr())

# We can look at the correlation between Netflix and the others
print(mult_df.corr()['NFLX'])

# We can plot this in a bar chart
mult_df.corr()['NFLX'].plot(kind='bar')

#plt.show()

# variance for NFLX
var = mult_df['NFLX'].var();
print()

days = len(mult_df.index)
print(days)

# true annual variance
print(days * var)

# co variance
co_var = mult_df.cov() * days
print(co_var)

# https://render.githubusercontent.com/view/ipynb?color_mode=dark&commit=2a0ee370a017a88ca620e7fb4bab24f0a6a6b73f&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f646572656b62616e61732f507974686f6e3446696e616e63652f326130656533373061303137613838636136323065376662346261623234663061366136623733662f507974686f6e253230666f7225323046696e616e6365253230332e6970796e62&nwo=derekbanas%2FPython4Finance&path=Python+for+Finance+3.ipynb&repository_id=397974623&repository_type=Repository#Why-do-We-Care-About-Risk
# Why do We Care About Risk
# Most investors don't handle massive flucuations in stock prices well. What we want to do at the very least is to make them aware of how dramatically their portfolios returns may be. We can then do our best to minimize risk by adding other stocks that have returns that aren't as closely correlated.

# Calculating a Portfolios Variance
# When calculating the variance of a portfolio we must define its weight, or how much of the portfolio it makes up. If you add up the weight of all stocks you get a value of 1.

portfolio_list = ['FB', 'NEM']

portfolio_df = merge_df_by_column_name('daily_return', '2020-1-1', '2020-12-31', *portfolio_list)

# not colerated
n_corr = portfolio_df.corr()
print(n_corr)

price_df = merge_df_by_column_name('Adj Close', '2020-1-1', '2020-12-31', *['FB', 'NEM'])

print(price_df)
# 253 days

fb_wt = 209.78 / 418.48 #<- portfolio value
nem_wt = 208.70 / 418.48 #<- portfolio value
# setting to 1
fb_wt = .5012
nem_wt = .4988

wts = np.array([fb_wt, nem_wt])

# tranpose rows into columns
port_var = np.dot(wts.T, np.dot(portfolio_df.cov() * 253, wts))
print("Portfolio Variance:", port_var)

print("FB Var:", portfolio_df['FB'].var() * 253)
print("NEM Var:", portfolio_df['NEM'].var() * 253)
