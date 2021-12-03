"""Get the stocks in main board.
"""
import os
import pandas as pd
import time

subdate_func = lambda x : \
    time.strftime("%Y-%m-%d", time.strptime(x, "%Y%m%d"))

print("=> Reading CSV file")
df = pd.read_csv("data/wind_daily_stock.csv")
print("=> Searching for main board stocks")
for val, main_df in df.groupby("class"):
    break
f = open("data/china_stock_qlib_adj/instruments/main.txt", "w")
print("=> Getting date range of stocks")
for code, stock_df in main_df.groupby("code"):
    symbol, ex = code.split(".")
    name = f"{ex}{symbol}"
    start_date, end_date = stock_df.date.min(), stock_df.date.max()
    start_date = subdate_func(str(start_date))
    end_date = subdate_func(str(end_date))
    f.write(f"{name}\t{start_date}\t{end_date}\n")
f.close()

    