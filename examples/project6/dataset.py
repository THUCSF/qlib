from torch.utils.data import Dataset


class TSDataset(Dataset):
  """Time series dataset (full enumeration).
  Args:
    df: The multivariate dataset. df[date][instrument]
    seq_len: The step length of data.
    horizon: The future length of label.
  """
  def __init__(self, df, seq_len=32, horizon=1):
    
    pass

