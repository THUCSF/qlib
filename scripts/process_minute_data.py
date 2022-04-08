"""Process a minute level CSV format data into qlib data.
Usage: in project root, run `python scripts/process_china_stock.py`
Make sure you have `data/SH000300_orig.csv` and `data/wind_daily_stock.csv`.
"""
import os
import pandas as pd
import time
import numpy as np


# process data into qlib-format
os.system("ipython scripts/dump_bin.py -- dump_all --csv_path data/china_stock_csv_adj --qlib_dir data/china_stock_qlib_adj --symbol_field_name code --include_fields open,high,low,close,volume,amount")
os.system("cp data/qlib_cn_stock/instruments/csi*.txt data/china_stock_qlib_adj/instruments")
# If this line results in error, please run copy the csi file from qlib's data to china_stock_qlib_adj manually.
# Or follow the instruction in README.md, download qlib data using:
# `ipython scripts/get_data.py -- qlib_data --target_dir data/qlib_cn_stock --region cn`