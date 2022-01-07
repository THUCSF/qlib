# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from __future__ import division
from __future__ import print_function

import torch, math, copy
import numpy as np
import pandas as pd
from typing import Text, Union

import torch.nn.functional as F
import pytorch_lightning as pl

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...log import get_module_logger


def collate_step_outputs(outs):
  """Collate the step outputs (y and pred) into an array.
  """
  N = len(outs)
  y, pred = [], []
  for i in range(N):
    pred.append(outs[i]["pred"])
    y.append(outs[i]["y"])
  return torch.cat(pred).squeeze(), torch.cat(y).squeeze()


class Learner(pl.LightningModule):
  def __init__(
    self,
    model,
    loss_type="rgr",
    lr=0.001,
    n_epoch=200,
    early_stop_patience=20,
    eval_n_epoch=20,
    weight_decay=0,
  ):
    super().__init__()
    self.model = model
    self.lr = lr
    self.n_epoch = n_epoch
    self.early_stop_patience = early_stop_patience
    self.eval_n_epoch = eval_n_epoch
    self.loss_type = loss_type
    self.weight_decay = weight_decay

    self.fitted = False
    self.model.to(self.device)

  def forward(self, batch):
    x, y = batch
    return self.model(x)

  def training_step(self, batch, batch_idx):
    x, y = batch
    pred = self.model(x)
    loss = self.loss_fn(pred, y)
    for param_group in self.optim.param_groups:
      self.log("lr", float(param_group['lr']))
      break
    return {"loss" : loss, "y": y, "pred": pred}

  def pred2score(self, pred):
    if "rgr" in self.loss_type:
      return pred.squeeze()
    elif "cls" in self.loss_type:
      prob = F.softmax(pred, 1)
      vals, pred_labels = prob.max(1)
      return vals + pred_labels
    elif "br" in self.loss_type:
      return pred[:, 0] - torch.sigmoid(pred[:, 1]) # mu - std: 85% chance lower bound

  def loss_fn(self, pred, label):
    loss = 0
    if "rgr" in self.loss_type:
      loss = loss + torch.square(pred - label).mean()
    elif "cls" in self.loss_type:
      loss = loss + F.cross_entropy(pred, label[:, 1].long())
    elif "br" in self.loss_type: # beyesian regression
      pred_y, pred_std = pred[:, 0], torch.sigmoid(pred[:, 1])
      pred_std = pred_std.clamp(min=1e-3)
      pred_sigma = pred_std ** 2
      const = math.log(2 * math.pi) / 2
      loss = (torch.square(pred_y - label) / pred_sigma).mean() / 2 \
        + torch.log(pred_std).mean() + const 
    return loss

  def predict_dataloader(self, dl, index):
    """Not applicable to time series.
    """
    self.model.eval()
    scores = []
    with torch.no_grad():
      for x in dl:
        pred = self.pred2score(self.model(x))
        scores.append(pred.detach().cpu().numpy())
    return pd.Series(np.concatenate(scores), index=index)

  def predict_dataframe(self, df):
    """Not applicable to time series.
    """
    index = df.index
    self.model.eval()
    x_values = df.values
    sample_num = x_values.shape[0]
    scores = []
    BS = 4096
    with torch.no_grad():
      for begin in range(sample_num)[::BS]:
        end = min(sample_num, begin + BS)
        x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
        pred = self.pred2score(self.model(x_batch))
        scores.append(pred.detach().cpu().numpy())
    return pd.Series(np.concatenate(scores), index=index)

  def predict(self, dataset: DatasetH, segment: Union[Text, slice] = "test"):
    x_test = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
    index = x_test.index
    self.model.eval()
    x_values = x_test.values
    sample_num = x_values.shape[0]
    scores = []
    BS = 4096
    with torch.no_grad():
      for begin in range(sample_num)[::BS]:
        end = min(sample_num, begin + BS)
        x_batch = torch.from_numpy(x_values[begin:end]).float().to(self.device)
        pred = self.pred2score(self.model(x_batch))
        scores.append(pred.detach().cpu().numpy())
    return pd.Series(np.concatenate(scores), index=index)

  def configure_optimizers(self):
    self.optim = torch.optim.AdamW(self.model.parameters(),
      lr=self.lr, weight_decay=self.weight_decay)
    self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, mode="max")
    return {"optimizer": self.optim, "lr_scheduler": self.sched, "monitor": "val_metric"}

  def _calc_high_rank_return(self, pred, y):
    score = self.pred2score(pred)
    pmin, pmax = score.min(), score.max()
    psmin = pmax - (pmax - pmin) * 0.1
    if "cls" in self.loss_type:
      ret = y[:, 0]
      mask = torch.isfinite(ret) & (score > psmin)
      hrr = ret[mask].mean() * 100 # shown in percentage
    else:
      mask = torch.isfinite(y) & (score > psmin)
      hrr = y[mask].mean() * 100 # shown in percentage
    return hrr
  
  def _calc_acc(self, pred, y):
    tp = (pred.argmax(1) == y).sum()
    acc = tp / y.shape[0]
    return acc

  def _calc_metric(self, outs):
    pred, y = collate_step_outputs(outs)
    return self._calc_high_rank_return(pred, y)

  def training_epoch_end(self, outs):
    res = self._calc_metric(outs)
    self.log("train_metric", res)

  def training_step_end(self, outs):
    self.log("train_loss", outs["loss"])
    return outs

  def validation_step(self, batch, batch_idx):
    x, y = batch
    pred = self.model(x)
    return {"y": y, "pred": pred}

  def validation_step_end(self, outs):
    return outs

  def validation_epoch_end(self, outs):
    hrr = self._calc_metric(outs)
    self.log("val_metric", hrr)
    return hrr


class MLP(torch.nn.Module):
  def __init__(self, input_dim, output_dim,
    hidden_dims=[256, 512, 768, 512, 256, 128, 64],
    dropout=-1):
    super(MLP, self).__init__()
    dims = [input_dim] + hidden_dims
    self.layers = torch.nn.ModuleList()
    for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
      self.layers.append(torch.nn.Linear(in_dim, out_dim))
      self.layers.append(torch.nn.ReLU(inplace=True))
      self.layers.append(torch.nn.BatchNorm1d(out_dim))
      if dropout > 0:
        self.layers.append(torch.nn.Dropout(dropout))
    self.layers.append(torch.nn.Linear(hidden_dims[-1], output_dim))

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x