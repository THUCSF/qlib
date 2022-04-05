"""Dataset"""
# pylint: disable=invalid-name
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


def calc_sample_indice(df, req_len, time_index, inst_index):
    trade_dates = df[time_index].unique() # array
    trade_dates.sort()
    num = len(trade_dates) - req_len + 1
    sample_indice = [[] for _ in range(num)]
    for _, inst in tqdm(df.groupby(inst_index)):
        inst_date = inst[time_index]
        for i in range(num):
            j = inst_date.searchsorted(trade_dates[i])
            if len(inst_date) <= j + req_len or \
                inst_date.iloc[j] != trade_dates[i] or \
                inst_date.iloc[j+req_len] != trade_dates[i+req_len]:
                continue
            st, ed = inst.index[j], inst.index[j+req_len]
            sample_indice[i].append((st, ed))
    while len(sample_indice[-1]) == 0:
        del sample_indice[-1]
        del trade_dates[-1]
    num = len(sample_indice)
    return trade_dates[-num:], sample_indice


class AlignedTSTensorDataset(Dataset):
    def __init__(self, df, data, sample_indice, horizon,
            input_names, target_names, time_index, inst_index):
        self.df, self.data = df, data
        self.sample_indice = sample_indice
        self.input_names = input_names
        self.target_names = target_names
        self.time_index, self.inst_index = time_index, inst_index
        self.horizon = horizon

    def __len__(self):
        return len(self.sample_indice)

    def describe(self):
        n_episodes = sum([len(day_indice) for day_indice in self.sample_indice])
        idx1 = self.sample_indice[0][0][1]
        idx2 = self.sample_indice[-1][0][1]
        st_date = self.df.iloc[idx1][self.time_index]
        ed_date = self.df.iloc[idx2][self.time_index]
        print(f"=> {st_date}-{ed_date}, {len(self.sample_indice)} days, {n_episodes} episodes.")

    def __getitem__(self, day_index):
        day_indice = self.sample_indice[day_index]
        st_idx, ed_idx = day_indice[0]
        target_len = len(self.target_names)
        x = torch.stack([self.data[st:ed, :-target_len]
                    for st, ed in day_indice], 1)
        y = torch.stack([self.data[st:ed, -target_len:]
                    for st, ed in day_indice], 1)
        return {
            "input": x,
            "input_time_start": str(self.df.iloc[st_idx][self.time_index]),
            "input_time_end": str(self.df.iloc[ed_idx][self.time_index]),
            "target": y}


class AlignedTSDataset(object):
    """Time series dataset where each batch has the same target time.
    """
    def __init__(self, df, seq_len=64, horizon=1,
                 time_index="datetime", inst_index="instrument",
                 input_names=None, target_names=None):
        self.df = df
        self.data = torch.from_numpy(df[input_names + target_names].values)
        self.time_index = time_index
        self.inst_index = inst_index
        self.input_names = input_names
        self.target_names = target_names
        self.horizon = horizon
        self.trade_dates, self.sample_indice = calc_sample_indice(
            df, seq_len, time_index, inst_index)
        self.trade_dates = pd.Series(self.trade_dates)

    def get_split(self, start_date, end_date):
        st = self.trade_dates.searchsorted(start_date)
        ed = self.trade_dates.searchsorted(end_date)
        return AlignedTSTensorDataset(
                self.df, self.data, self.sample_indice[st:ed],
                self.horizon, self.input_names, self.target_names,
                self.time_index, self.inst_index)


class TSDataset(Dataset):
    """Time series dataset (full enumeration).
    Args:
        df: The multivariate dataset. df[date][instrument]
        seq_len: The step length of data.
        horizon: The future length of label.
        TIME_AXIS: The name of time index.
        INST_AXIS: The name of instance of series.
    """
    def __init__(self, df, seq_len=64, horizon=1,
                 time_index="datetime", inst_index="instrument",
                 target_names=None, input_names=None):
        self.df = df
        self.data = torch.from_numpy(df[input_names + target_names].values)
        self.time_index, self.inst_index = time_index, inst_index
        self.target_names = target_names
        self.input_names = input_names
        self.seq_len = seq_len
        self.horizon = horizon
        count = df.groupby(inst_index).datetime.count()  # dataframe
        start_indice = df.groupby(inst_index).apply(lambda x: x.index[0])
        self.INST = pd.DataFrame({"count": count, "start": start_indice})
        indice = []
        for idx in range(self.INST.shape[0]):
            L = self.INST.iloc[idx]["count"]
            g_idx = self.INST.iloc[idx]["start"]
            if L >= seq_len:
                indice.extend([(g_idx + offset, g_idx + offset + seq_len)
                    for offset in range(L - seq_len + 1)])
        self.sample_indice = np.array(indice)

    def describe(self):
        print(f"=> {len(self.sample_indice)} episodes.")

    def instance_group(self):
        """Group by instance."""
        return self.df.groupby(self.inst_index)

    def __len__(self):
        return self.sample_indice.shape[0]

    def __getitem__(self, index):
        """Note that the DataLoader will put the batch dimension on 0-axis."""
        st, ed = self.sample_indice[index]
        target_len = len(self.target_names)
        x = self.data[st:ed, :-target_len]
        y = self.data[st:ed, -target_len:]

        return {
            "input": x,
            "input_time_start": str(self.df.iloc[st][self.time_index]),
            "input_time_end": str(self.df.iloc[ed][self.time_index]),
            "target": y}
