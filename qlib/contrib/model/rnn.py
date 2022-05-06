from re import A
import torch
import math
import pandas as pd
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from typing import Text, Union

from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP


def collate_step_outputs(outs):
    """Collate the step outputs (y and pred) into an array."""
    N = len(outs)
    y, pred = [], []
    for i in range(N):
        pred.append(outs[i]["pred"])
        y.append(outs[i]["y"])
    return torch.cat(pred).squeeze(), torch.cat(y).squeeze()


class QuantileLoss(torch.autograd.Function):
    """Calculate the quantile loss between two sequences.
    Differentiable loss of sorting in quantile.
    Args:
        coef: A scaling parameter.
        n_q: The number of quantiles.
        t_q: The threshold of quantile difference to be minimized.
    """

    def __init__(self, coef=1.0, n_q=10, t_q=1):
        super().__init__()
        self.coef = coef
        self.n_q, self.t_q = n_q, t_q

    @staticmethod
    def forward(ctx, dt_seq : torch.Tensor, gt_seq : torch.Tensor, coef : float = 1.0, n_q : int = 10, t_q : int = 1):
        """
        Args:
            dt_seq: Shape (N,) Tensor, the sequence to be differentiated.
            gt_seq: Shape (N,) Tensor, the groundtruth values.
        """
        dt_idx = dt_seq.argsort() # in ascenting order
        gt_idx = gt_seq.argsort()
        dt_q = torch.zeros_like(dt_idx)
        gt_q = torch.zeros_like(gt_idx)
        step_size = math.floor(dt_idx.shape[0] / float(n_q))
        for q_idx in range(n_q):
            st, ed = step_size * q_idx, step_size * (q_idx + 1)
            ed = min(dt_idx.shape[0], ed)
            dt_q[dt_idx[st:ed]] = q_idx
            gt_q[gt_idx[st:ed]] = q_idx
        diff_q = (gt_q - dt_q).float()
        ctx.save_for_backward(diff_q)
        ctx.coef, ctx.t_q = coef, t_q 
        return diff_q.abs().mean()

    @staticmethod
    def backward(ctx, grad_output):
        """
        Args:
            grad_output: The gradient from above.
        """
        coef, t_q = ctx.coef, ctx.t_q
        diff_q, = ctx.saved_tensors
        mask = diff_q.abs() > t_q
        grad = (mask * diff_q).float()
        return coef * grad * grad_output, -coef * grad * grad_output, None, None, None


def quantile_loss(dt_seq, gt_seq, coef=1.0, n_q=10, t_q=1):
    return QuantileLoss.apply(dt_seq, gt_seq, coef, n_q, t_q)


class RNNLearner(pl.LightningModule):
    def __init__(
        self,
        model,
        loss_type="rgr-last",
        lr=0.001,
        weight_decay=1e-4,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_type = loss_type
        self.weight_decay = weight_decay

        self.fitted = False
        self.model.to(self.device)

    def forward(self, batch):
        input = batch["input"].cuda()
        info = {
            "pred": self.model(input),
            "target_time_start": batch["target_time_start"],
            "target_time_end": batch["target_time_end"],
            "instance": batch["instance"],
        }
        if "target" in batch:
            info["gt"] = batch["target"]
        return info

    def training_step(self, batch, batch_idx):
        input, target = batch["input"].cuda(), batch["target"].cuda()
        if input.shape[0] == 1 and len(input.shape) == 4:
            # squeeze if there is an extra batch dimension
            input = input[0]
            target = target[0]
        # 64, 123, 1
        if "last" in self.loss_type:
            pred = self.model(input)
            loss = self.loss_fn(pred, target[-1])
        else:
            pred = self.model(input, last_only=False)
            loss = self.loss_fn(pred, target)
            pred = pred[-1]
        target = target[-1]
        for param_group in self.optim.param_groups:
            self.log("lr", float(param_group["lr"]))
            break
        return {"loss": loss, "y": target.detach().cpu(), "pred": pred.detach().cpu()}

    def pred2score(self, pred):
        if "rgr" in self.loss_type or "quantile" in self.loss_type:
            return pred.squeeze()
        elif "cls" in self.loss_type:
            prob = F.softmax(pred, 1)
            vals, pred_labels = prob.max(1)
            return vals + pred_labels
        elif "br" in self.loss_type:
            # mu - std: 85% chance lower bound
            return pred[-1, :, 0] - torch.exp(pred[-1, :, 1] / 2)

    def loss_fn(self, pred, label):
        loss = 0
        if "rgr" in self.loss_type:
            #loss = loss + torch.square(pred - label).mean()
            nomin = (pred - label).abs()
            denom = pred.abs().detach() + label.abs()
            loss = loss + (nomin / denom).mean() * 100
        elif "quantile" in self.loss_type:
            loss = loss + quantile_loss(pred, label)
        elif "br" in self.loss_type:  # beyesian regression
            pred_y, pred_logvar = pred[:, :, :1], pred[:, :, 1:]
            pred_var = torch.exp(pred_logvar)
            const = math.log(2 * math.pi) / 2
            loss = (
                (torch.square(pred_y - label) / pred_var).mean() / 2
                + 0.5 * torch.log(pred_var).mean()
                + const
            )
        # elif "cls" in self.loss_type:
        #    loss = loss + F.cross_entropy(pred, label[:, 1].long())
        return loss

    def configure_optimizers(self):
        self.optim = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optim, T_max=5, eta_min=1e-5
        )
        return {
            "optimizer": self.optim,
            "lr_scheduler": self.sched,
            "monitor": "train_loss",
        }

    def predict_dataset(self, ds):
        """Predict scores from a dataset. Need the sample indice from ds."""
        self.model.eval()
        preds, scores = [], []
        BS = 1024
        # pinned inputs
        x = torch.Tensor(ds.seq_len, BS, len(ds.input_names)).cuda()
        values = torch.from_numpy(ds.df[ds.input_names].values)
        with torch.no_grad():
            for idx in tqdm(range(len(ds))):
                st, ed = ds.sample_indice[idx]
                x[:, idx % BS].copy_(values[st:ed], True)
                if (idx + 1) % BS == 0:
                    preds.append(self.model(x).squeeze())
                    scores.append(self.pred2score(preds[-1]))
            if len(ds) % BS != 0:
                preds.append(self.model(x[:, : len(ds) % BS]).squeeze())
                scores.append(self.pred2score(preds[-1]))
        # QLib uses current signal to buy the next bar's property
        target_indice = ds.sample_indice[:, 1] + ds.horizon - 2  # (N,)
        insts = ds.df[ds.inst_index].values[target_indice]
        dates = ds.df[ds.time_index].values[target_indice]
        scores = torch.cat(scores).cpu().numpy()
        preds = torch.cat(preds).cpu().numpy()
        return scores, preds, insts, dates, target_indice

    def _calc_acc(self, pred, y):
        tp = (pred.argmax(1) == y).sum()
        acc = tp / y.shape[0]
        return acc

    def _calc_metric(self, outs):
        preds, ys = [], []
        for out in outs:
            pred = out["pred"].squeeze().cpu().numpy()
            y = out["y"].squeeze().cpu().numpy()
            if len(y.shape) == 0:
                return math.nan
            pred_indice = pred.argsort()
            gt_indice = y.argsort()
            Q = min(50, int(pred.shape[0] * 0.1))
            preds.append(y[pred_indice[-Q:]].mean())
            ys.append(y[gt_indice[-Q:]].mean())
        preds = np.array(preds).astype("float32")
        ys = np.array(ys).astype("float32")
        return (ys - preds).mean()

    def training_epoch_end(self, outs):
        res = self._calc_metric(outs)
        self.log("train_metric", res)

    def training_step_end(self, outs):
        self.log("train_loss", outs["loss"])
        return outs

    def validation_step(self, batch, batch_idx):
        input, target = batch["input"].cuda(), batch["target"].cuda()
        if input.shape[0] == 1 and len(input.shape) == 4:
            input = input[0]
            target = target[0]
        pred = self.model(input)
        return {"y": target[-1].detach().cpu(), "pred": pred.detach().cpu()}

    def validation_step_end(self, outs):
        return outs

    def validation_epoch_end(self, outs):
        res = self._calc_metric(outs)
        self.log("val_metric", res)
        return res


class RNN(torch.nn.Module):
    def __init__(self, core_type="LSTM", output_size=1, **kwargs):
        super().__init__()
        self.core_type = core_type
        if core_type == "LSTM":
            self.core = torch.nn.LSTM(**kwargs)
            self.fc_out = torch.nn.Linear(kwargs["hidden_size"], output_size)
        elif core_type == "MultiStreamLSTM":
            self.cores = torch.nn.ModuleList(
                [torch.nn.LSTM(**kwargs) for _ in range(output_size)]
            )
            self.fc_outs = torch.nn.ModuleList(
                [torch.nn.Linear(kwargs["hidden_size"], 1) for _ in range(output_size)]
            )

    def forward(self, x, last_only=True):
        if self.core_type == "LSTM":
            out, _ = self.core(x)
            if last_only:
                return self.fc_out(out[-1])
            return self.fc_out(out)
        elif self.core_type == "MultiStreamLSTM":
            pred = []
            for core, fc_out in zip(self.cores, self.fc_outs):
                out, _ = core(x)
                if last_only:
                    pred.append(fc_out(out[-1]))
                else:
                    out = fc_out(out.view(-1, out.shape[-1]))
                    pred.append(out.view(x.shape[0], x.shape[1], -1))
            pred = torch.cat(pred, -1)
            return pred
