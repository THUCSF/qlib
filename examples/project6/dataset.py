"""Dataset"""
# pylint: disable=invalid-name
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


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
