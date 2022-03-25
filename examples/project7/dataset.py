"""Dataset"""
# pylint: disable=invalid-name
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class AlignedTSDataset(Dataset):
    """Time series dataset where each batch has the same target time.
    """
    def __init__(self, df, seq_len=64, horizon=1,
                 time_index="datetime", inst_index="instrument",
                 target_names=None, input_names=None):
        self.df = df
        self.data = torch.from_numpy(self.df[input_names + target_names].values)
        self.time_index, self.inst_index = time_index, inst_index
        self.target_names = target_names
        self.input_names = input_names
        self.seq_len = seq_len
        self.horizon = horizon
        self.trade_dates = df[time_index].unique() # array
        self.trade_dates.sort()
        req_len = seq_len + horizon # required data length
        self.batch_num = len(self.trade_dates) - req_len
        self.sample_indice = [[] for _ in range(self.batch_num)]
        for _, inst in df.groupby(inst_index):
            inst_date = inst[time_index]
            for i in range(self.batch_num):
                j = inst_date.searchsorted(self.trade_dates[i])
                if len(inst_date) <= j + req_len or \
                    inst_date.iloc[j] != self.trade_dates[i] or \
                    inst_date.iloc[j+req_len] != self.trade_dates[i+req_len]:
                    continue
                st, ed = inst.index[j], inst.index[j+req_len]
                self.sample_indice[i].append((st, ed))

    def describe(self):
        n_episodes = sum([len(day_indice) for day_indice in self.sample_indice])
        print(f"=> {len(self.sample_indice)} Days, {n_episodes} episodes.")

    def __len__(self):
        return len(self.sample_indice)

    def __getitem__(self, day_index):
        day_indice = self.sample_indice[day_index]
        start_index, end_index = day_indice[0]
        target_len = len(self.target_names)
        x = torch.stack([self.data[st:ed-self.horizon, :-target_len]
            for st, ed in day_indice])
        y = torch.stack([self.data[ed-self.horizon:ed, -target_len:]
            for _, ed in day_indice])
        return {
            "input": x,
            "input_time_start": str(self.df.iloc[start_index][self.time_index]),
            "input_time_end": str(self.df.iloc[end_index][self.time_index]),
            "target": y,
            "day_index": day_index}


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
            if L > seq_len - horizon:
                indice.extend([(g_idx + offset, g_idx + offset + seq_len)
                    for offset in range(L - seq_len - horizon)])
        self.sample_indice = np.array(indice)

    def describe(self):
        print(f"=> {len(self.sample_indice)} episodes.")

    def instance_group(self):
        """Group by instance."""
        return self.df.groupby(self.inst_index)

    def __len__(self):
        return self.sample_indice.shape[0]

    def __getitem__(self, index):
        st, ed = self.sample_indice[index]
        target_len = len(self.target_names)
        x = self.data[st:ed, :-target_len]
        y = self.data[ed:ed+self.horizon, -target_len:]

        return {
            "input": x,
            "input_time_start": str(self.df.iloc[st][self.time_index]),
            "input_time_end": str(self.df.iloc[ed][self.time_index]),
            "target": y}
