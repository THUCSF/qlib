"""Process a minute level CSV format data and resample it into 30-min frequency.
"""
import os
from multiprocessing import Pool, Lock
from zipfile import ZipFile
from tqdm import tqdm
import pandas as pd

TEMP_DIR = "data/temp_csv"  # storing temp csv
OUT_DIR = "data/china_stock_30min_csv"
WIN=30
TOTAL=240

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

zfile = ZipFile("data/gettinydata.zip")
fileinfos = list(
    filter(lambda x: ".csv" in x.filename and x.file_size > 32, zfile.filelist)
)
zipfile_lock = Lock()
def worker(finfo):
    """Process an item in zipfile."""
    zipfile_lock.acquire()
    zfile.extract(finfo, TEMP_DIR)
    zipfile_lock.release()
    N_WIN = TOTAL // WIN
    csv_path = os.path.join(TEMP_DIR, finfo.filename)
    data = pd.read_csv(csv_path)
    price_base = data.open[::TOTAL].values
    price_base = price_base[None, :].repeat(N_WIN, 1).reshape(-1)
    lows = data.low.values.reshape(30, -1).min(0)
    highs = data.high.values.reshape(30, -1).max(0)
    closes = data.close[WIN-1::WIN].values
    opens = data.open[::WIN].values
    vols = data.vol.values.reshape(30, -1).sum(0)
    amounts = data.amount.values.reshape(30, -1).sum(0)
    lows, highs = lows - price_base, highs - price_base
    opens, closes = opens - price_base, closes - price_base
    names = ["open", "close", "high", "low", "vol", "amount"]
    arrays = [opens, closes, highs, lows, vols, amounts]
    new_date = data.date[TOTAL-1::TOTAL].values
    new_code = data.StockID[TOTAL-1::TOTAL].values
    new_data = pd.DataFrame({"date": new_date, "StockID": new_code}, )
    for name, array in zip(names, arrays):
        for i in range(N_WIN):
            item_name = f"{name}_{WIN}min_{i+1}"
            new_data[item_name] = array[i::N_WIN]
    fname = finfo.filename[finfo.filename.rfind("/")+1:]
    new_data.to_csv(f"{OUT_DIR}/{fname}", float_format="%.2f")
    os.system(f"rm {csv_path}")
    print(f"=> Stage 1: {OUT_DIR}/{fname} processed")

def worker_append(finfo):
    """Process an item in zipfile."""
    zipfile_lock.acquire()
    zfile.extract(finfo, TEMP_DIR)
    zipfile_lock.release()
    N_WIN = TOTAL // WIN
    csv_path = os.path.join(TEMP_DIR, finfo.filename)
    data = pd.read_csv(csv_path)
    price_base = data.open[::TOTAL].values
    price_base = price_base[None, :].repeat(N_WIN, 1).reshape(-1)
    lows = data.low.values.reshape(30, -1).min(0)
    highs = data.high.values.reshape(30, -1).max(0)
    closes = data.close[WIN-1::WIN].values
    opens = data.open[::WIN].values
    vols = data.vol.values.reshape(30, -1).sum(0)
    amounts = data.amount.values.reshape(30, -1).sum(0)
    lows, highs = lows - price_base, highs - price_base
    opens, closes = opens - price_base, closes - price_base
    names = ["open", "close", "high", "low", "vol", "amount"]
    arrays = [opens, closes, highs, lows, vols, amounts]
    new_date = data.date[TOTAL-1::TOTAL].values
    new_data = pd.DataFrame({"date": new_date})
    for name, array in zip(names, arrays):
        for i in range(N_WIN):
            item_name = f"{name}_{WIN}min_{i+1}"
            new_data[item_name] = array[i::N_WIN]
    fname = finfo.filename[finfo.filename.rfind("/")+1:]
    old_data = pd.read_csv(f"{OUT_DIR}/{fname}")
    new_data = old_data.append(new_data)
    new_data.to_csv(f"{OUT_DIR}/{fname}", float_format="%.2f")
    os.system(f"rm {csv_path}")
    print(f"=> Stage 2: {OUT_DIR}/{fname} processed")

#with Pool(4) as mpool:
#    mpool.map(worker, fileinfos, chunksize=1)
zfile = ZipFile("data/data_min.zip")
fileinfos = list(
    filter(lambda x: ".csv" in x.filename and x.file_size > 32, zfile.filelist)
)
with Pool(4) as mpool:
    mpool.map(worker_append, fileinfos, chunksize=1)