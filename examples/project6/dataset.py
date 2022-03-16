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
        self.data = self.df[target_names + input_names].values
        self.time_index, self.inst_index = time_index, inst_index
        self.target_names = target_names
        self.input_names = input_names
        self.seq_len = seq_len
        self.horizon = horizon
        self.trade_dates = df[time_index].unique() # array
        req_len = seq_len + horizon # required data length
        self.batch_num = len(self.trade_dates) - req_len
        self.sample_indice = [[] for _ in range(self.batch_num)]
        for _, inst in df.groupby(inst_index):
            for i in range(self.batch_num):
                start_date = self.trade_dates[i]
                end_date = self.trade_dates[i + req_len]
                valid_mask = (inst.datetime >= start_date) & \
                    (inst.datetime < end_date)
                if valid_mask.sum() < req_len:
                    continue # there is missing data
                index = inst[valid_mask].index
                self.sample_indice[i].append((index[0], index[-1] + 1))

    def __len__(self):
        return len(self.sample_indice)

    def __getitem__(self, day_index):
        day_indice = self.sample_indice[day_index]
        start_index, end_index = day_indice[0]

        x = torch.cat([torch.from_numpy(self.data
                [st:ed-self.horizon, :-1]) for st, ed in day_indice])
        y = torch.cat([torch.from_numpy(self.data
                [ed-self.horizon:ed, -1]) for _, ed in day_indice])

        return {
            "input": x,
            "input_time_start": str(self.df.iloc[start_index][self.time_index]),
            "input_time_end": str(self.df.iloc[end_index][self.time_index]),
            "target": y}


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
        #self.rng = np.random.RandomState(seed)
        #self.random_index = np.arange(self.sample_indice.shape[0])
        # self.rng.shuffle(self.random_index)

    def instance_group(self):
        """Group by instance."""
        return self.df.groupby(self.inst_index)

    def __len__(self):
        return self.sample_indice.shape[0]

    def __getitem__(self, index):
        st, ed = self.sample_indice[index]
        input_arr = self.df.iloc[st:ed]
        target_arr = self.df.iloc[ed:ed+self.horizon]

        input_data = torch.from_numpy(input_arr[self.input_names].values)
        target = torch.from_numpy(target_arr[self.target_names].values)

        return {
            "input": input_data,
            "input_time_start": str(input_arr[self.time_index].values[0]),
            "input_time_end": str(input_arr[self.time_index].values[-1]),
            "target": target,
            "target_time_start": str(target_arr[self.time_index].values[0]),
            "target_time_end": str(target_arr[self.time_index].values[-1]),
            "instance": input_arr[self.inst_index].values[0]}
