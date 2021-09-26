import numpy as np
import pandas as pd

from pandas_datareader import data as web
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import mplfinance as mpf

def save_to_csv_from_yahoo(ticker, syear, smonth, sday, eyear, emonth, eday):
    start = dt.datetime(syear, smonth, sday)
    end = dt.datetime(eyear, emonth, eday)
    df = web.DataReader(ticker, 'yahoo', start, end)
    df.to_csv("C:\\DEV\\Python\\Stock_Sandbox\\stockdata\\" + ticker + ".csv")

    return df

def get_df_from_csv(ticker):
    try:
        df = pd.read_csv("C:\\DEV\\Python\\Stock_Sandbox\\stockdata\\" + ticker + ".csv")
    except FileNotFoundError:
        print("File not found for ticker: {}".format(ticker))
    else:
        return df

def add_daily_return_to_df(df, ticker):
    df['daily_return'] = (df['Adj Close'] / df['Adj Close'].shift(1)) - 1
    df.to_csv("C:\\DEV\\Python\\Stock_Sandbox\\stockdata\\" + ticker + ".csv")
    return

def get_return_defined_time(df, syear, smonth, sday, eyear, emonth, eday):
    start = f"{syear}-{smonth}-{sday}"
    end = f"{eyear}-{emonth}-{eday}"
    df['Date'] = pd.to_datetime(df['Date'])
    mask = (df['Date'] >= start) & (df['Date'] <= end)
    daily_ret = df.loc[mask]['daily_return'].mean()
    df2 = df.loc[mask]
    days = df2.shape[0]
    return (days * daily_ret)

def mplfinance_plot(ticker, chart_type, syear, smonth, sday, eyear, emonth, eday):
    start = f"{syear}-{smonth}-{sday}"
    end = f"{eyear}-{emonth}-{eday}"
    try:
        df = pd.read_csv("C:\\DEV\\Python\\Stock_Sandbox\\stockdata\\" + ticker + ".csv")
    except FileNotFoundError:
        print("File not found for ticker: {}".format(ticker))
    else:
        df.index = pd.DatetimeIndex(df['Date'])
        df_sub = df.loc[start:end]
        mpf.plot(df_sub, type='candle')
        mpf.plot(df_sub, type='line')
        mpf.plot(df_sub, type='ohlc', mav=4) # moving average

        s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 8})
        fig = mpf.figure(figsize=(12, 8), style=s)

        ax = fig.add_subplot(2,1,1) 
        av = fig.add_subplot(2,1,2, sharex=ax) 

        mpf.plot(df_sub,type=chart_type, mav=(3,5,7), ax=ax, volume=av, show_nontrading=True)

def price_plot(ticker, syear, smonth, sday, eyear, emonth, eday):
    # Create string representations for the dates
    start = f"{syear}-{smonth}-{sday}"
    end = f"{eyear}-{emonth}-{eday}"
    
    try:
         df = pd.read_csv("C:\\DEV\\Python\\Stock_Sandbox\\stockdata\\" + ticker + ".csv")
    except FileNotFoundError:
        print("File not found for ticker: {}".format(ticker))
    else:
        df.index = pd.DatetimeIndex(df['Date'])
        df_sub = df.loc[start:end]
        df_np = df_sub.to_numpy()

        np_adj_close = df_np[:,5]

        date_arr = df_np[:,1]
        
        fig = plt.figure(figsize=(12,8))
        axes = fig.add_axes([0,0,1,1])
        
        axes.plot(date_arr, np_adj_close, color='navy')
        
        axes.xaxis.set_major_locator(plt.MaxNLocator(8))
        
        axes.grid(True, color='0.6', dashes=(5, 2, 1, 2))
        
        axes.set_facecolor('#FAEBD7')

        plt.show()

def download_multiple_stocks(syear, smonth, sday, eyear, emonth, eday, *args):
    for x in args:
        save_to_csv_from_yahoo(x, syear, smonth, sday, eyear, emonth, eday)

def merge_df_by_column_name(col_name, syear, smonth, sday, eyear, emonth, eday, *tickers):
    mult_df = pd.DataFrame()
    start = f"{syear}-{smonth}-{sday}"
    end = f"{eyear}-{emonth}-{eday}"

    for ticker in tickers:
        mult_df[ticker] = web.DataReader(ticker, 'yahoo', start, end)[col_name]

    return mult_df

def plot_return_mult_stocks(investment, stock_df):
    (stock_df/ stock_df.iloc[0] * investment).plot(figsize=(15,6))
    plt.show()

def standard_deviation_implementation(elements):
    mean = np.mean(elements)
    sum = 0

    for e in elements:
        sum += (e - mean) ** 2

    return np.sqrt(sum / len(elements))

def get_stock_mean_sd(stock_df, ticker):
    return stock_df[ticker].mean(), stock_df[ticker].std()

def get_mult_stock_mean_sd(stock_df):
    for stock in stock_df:
        mean, sd = get_stock_mean_sd(stock_df, stock)
        cov = sd / mean
        print("Stock: {:4} Mean: {:7.2f} Standard Deviation {:2.2f}"
        .format(stock, mean, sd))
        print("Coefficent of Variation: {}\n".format(cov))

# save_to_csv_from_yahoo('AMZN', 2020, 1, 1, 2021, 1, 1)
# AMZN = get_df_from_csv("AMZN")

# add_daily_return_to_df(AMZN, 'AMZN')
# print(AMZN)

# total_ret = get_return_defined_time(AMZN, 2020, 1, 1, 2021, 1, 1)

# print("Total Return: ", total_ret)

# mplfinance_plot('AMZN', 'ohlc', 2020, 1, 1, 2021, 1, 1)

# price_plot('AMZN', 2020, 1, 1, 2021, 1, 1)

# tickers = ['FB', 'AAPL', 'NFLX', 'GOOG']
# download_multiple_stocks(2020, 1, 1, 2021, 1, 1, *tickers)


tickers = ['FB', 'AMZN', 'AAPL', 'NFLX', 'GOOG']
multi_df = merge_df_by_column_name('Adj Close', 2020, 1, 1, 2021, 1, 1, tickers)

# place $100 in each stock at the start of 2020
# plot_return_mult_stocks(100, multi_df)

# values = [1,2,3,4,5,6]
# standard_deviation_implementation(values)

# print(np.std(values))

get_mult_stock_mean_sd(multi_df)
print(multi_df)