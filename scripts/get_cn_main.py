"""Get the stocks in main board.
"""
import glob
import time
from tqdm import tqdm
import pandas as pd


subdate_func = lambda x : \
    time.strftime("%Y-%m-%d", time.strptime(x, "%Y%m%d"))

csv_files = glob.glob("data/data_min/*.csv")
csv_files.sort()
f = open("data/china_stock_qlib_adj/instruments/main.txt", "w")
for code, stock_df in main_df.groupby("code"):
    symbol, ex = code.split(".")
    name = f"{ex}{symbol}"
    start_date, end_date = stock_df.date.min(), stock_df.date.max()
    start_date = subdate_func(str(start_date))
    end_date = subdate_func(str(end_date))
    f.write(f"{name}\t{start_date}\t{end_date}\n")
f.close()

    