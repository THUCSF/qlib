"""Process a CSV format data into qlib data.
Usage: in project root, run `python <path to this script>`
Make sure you have `data/SH000300_orig.csv` and `data/wind_daily_stock.csv`.
"""
import os
import pandas as pd
import time

print("=> Converting CSI300")
df = pd.read_csv("data/SH000300_orig.csv", index_col=False)
subdate_func = lambda x : \
    time.strftime("%Y%m%d", time.strptime(x, "%Y-%m-%d"))
df.date = df.date.apply(subdate_func)
df.set_index("date")
df.to_csv("data/SH000300.csv", index=False)

if not os.path.exists("data/china_stock_csv_adj"):
    os.makedirs("data/china_stock_csv_adj")
# copy CSI300 to the data directory
os.system("cp data/SH000300.csv data/china_stock_csv_adj")

print("=> Converting CSV file")
df = pd.read_csv("data/wind_daily_stock.csv")
for key in ["open", "high", "low", "close"]:
    df.drop(columns=key)
    df.rename({f"adj{key}" : key})
for symbol, cdf in df.groupby("code"):
    symbol, ex = symbol.split(".")
    name = f"{ex}{symbol}"
    cdf.code = symbol
    cdf.to_csv(f"data/china_stock_csv_adj/{name}.csv", index=False)

# process data into qlib-format
os.system("ipython scripts/dump_bin.py -- dump_all --csv_path data/china_stock_csv_adj --qlib_dir data/china_stock_qlib_adj --symbol_field_name code --include_fields open,high,low,close,volume,amount")