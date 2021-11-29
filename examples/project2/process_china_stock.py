import pandas as pd
import numpy as np

df = pd.read_csv("data/wind_daily_stock.csv")
for symbol, cdf in df.groupby("stck_code"):
    if symbol > 300000 and symbol < 500000: # starting with 300000
        EX = "sz"
    else:
        EX = "sh"
    name = f"{EX}{symbol:06d}"
    cdf.to_csv(f"data/china_stock/{name}.csv", index=False)

# python scripts/dump_bin.py dump_all --csv_path data/china_stock --qlib_dir data/cn_data_qlib --symbol_field_name stck_code --exclude_fields stck_code,company