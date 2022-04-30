"""Process a minute level CSV format data into qlib data.
Usage: in project root, run `python scripts/process_china_stock.py`
Make sure you have `data/SH000300_orig.csv` and `data/wind_daily_stock.csv`.
"""
import os
from threading import Thread
from zipfile import ZipFile
from tqdm import tqdm
from scripts.dump_bin import DumpDataUpdate, DumpDataAll


class GeneralThread(Thread):
    """Function interface threading."""

    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args, self.kwargs = args, kwargs

    def run(self):
        self.res = self.func(*self.args, **self.kwargs)


TEMP_DIR = "data/temp_csv"  # storing temp csv
QLIB_DATA_DIR = "data/minute_qlib"
INTERVAL = 100
STEP = 1

if STEP == 1:
    zfile = ZipFile("data/gettinydata.zip")
    fileinfos = list(
        filter(lambda x: ".csv" in x.filename and x.file_size > 32, zfile.filelist)
    )

    for idx, finfo in enumerate(tqdm(fileinfos)):
        zfile.extract(finfo, TEMP_DIR)
        if (idx + 1) % INTERVAL == 0 or idx == len(fileinfos) - 1:
            func_name = DumpDataAll if not os.path.exists(QLIB_DATA_DIR) else DumpDataUpdate
            csv_path = os.path.join(TEMP_DIR, "gettinydata")
            func = func_name(
                csv_path=csv_path,
                qlib_dir=QLIB_DATA_DIR,
                freq="minute",
                date_field_name="date",
                symbol_field_name="StockID",
                include_fields="open,high,low,close,vol,amount",
            )()
            os.system(f"rm -r {TEMP_DIR}")

if STEP == 2:
    zfile = ZipFile("data/data_min.zip")
    fileinfos = list(
        filter(lambda x: ".csv" in x.filename and x.file_size > 32, zfile.filelist)
    )

    for idx, finfo in enumerate(tqdm(fileinfos)):
        zfile.extract(finfo, TEMP_DIR)
        if (idx + 1) % INTERVAL == 0 or idx == len(fileinfos) - 1:
            csv_path = os.path.join(TEMP_DIR, "data_min")
            func = DumpDataUpdate(
                csv_path=csv_path,
                qlib_dir=QLIB_DATA_DIR,
                freq="minute",
                date_field_name="date",
                symbol_field_name="StockID",
                include_fields="open,high,low,close,vol,amount",
            )()
            os.system(f"rm -r {TEMP_DIR}")