import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from scipy.stats import pearsonr
import os
import sys

#Model/Training related libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.nn import Parameter
from functools import wraps

#Dataloader libraries
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from backtest import Backtest


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
deciles = [-8, -6, -4, -2, 0, 2, 4, 6, 8]

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers, bias=True, batch_first=True, dropout=0.5)
        self.Linear = nn.Linear(hidden_size, output_size)
        self._weight_init()

    def _weight_init(self):
        for name, param in self.rnn.named_parameters():
            if name.startswith("weight"):
                nn.init.uniform_(param, -1 / np.sqrt(self.hidden_size), 1 / np.sqrt(self.hidden_size)).to(device)   
        nn.init.kaiming_normal_(self.Linear.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")

    def forward(self, x, lengths):
        x = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, (h_n, c_n) = self.rnn(x)
        x, x_len = pad_packed_sequence(x, batch_first=True)
        x = self.Linear(x)
        return x, x_len


class TrainDataset(Dataset):
    # load the dataset
    def __init__(self, x, y, seq_len):#, context = 0):
        self.X = x
        self.Y = y
        self.seq_len = seq_len
        # self.use_saved_model = use_saved_model
        # self.period = period
        # self.context = context

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.Y)

    # get row item at some index
    def __getitem__(self, index):
        x = torch.FloatTensor(self.X[index])
        y = torch.FloatTensor(self.Y[index])
        return x, y

    def collate_fn(batch):
        batch_x = []
        batch_y = []
        len_x = []
        len_y = []
        for b in range(len(batch)):
            x, y = batch[b]
            length = 64
            index = np.random.randint(0, len(x) - length)
            batch_x.append(x[index: index + length])
            batch_y.append(y[index: index + length])
            # len_x.append(length)
            # len_y.append(length)

        len_x = torch.LongTensor([len(x) for x in batch_x])
        len_y = torch.LongTensor([len(y) for y in batch_y])

        pad_x = pad_sequence(batch_x, batch_first=True)
        pad_y = pad_sequence(batch_y, batch_first=True)
        return pad_x, pad_y, len_x, len_y


# class Weighted_TrainDataset(Dataset):
#     # load the dataset
#     def __init__(self, x, y, seq_len):#, use_saved_model, period):#, context = 0):
#         self.X = x
#         self.Y = y
#         self.seq_len = seq_len
#         # self.use_saved_model = use_saved_model
#         # self.period = period
#         # self.context = context

#     # get number of items/rows in dataset
#     def __len__(self):
#         return len(self.Y)

#     # get row item at some index
#     def __getitem__(self, index):
#         x = torch.FloatTensor(self.X[index])
#         y = torch.FloatTensor(self.Y[index])
#         return x, y     


class TestDataset(Dataset):
    # load the dataset
    def __init__(self, x, y, seq_len):#, context = 0):
        self.X = x
        self.Y = y
        self.seq_len = seq_len
        # self.context = context

    # get number of items/rows in dataset
    def __len__(self):
        return len(self.Y)

    # get row item at some index
    def __getitem__(self, index):
        x = torch.FloatTensor(self.X[index])
        y = torch.FloatTensor(self.Y[index])
        return x, y

    def collate_fn(batch):
        batch_x = [x for x, y in batch]
        batch_y = [y for x, y in batch]
        len_x = torch.LongTensor([len(x) for x in batch_x])
        len_y = torch.LongTensor([len(y) for y in batch_y])
        pad_x = pad_sequence(batch_x, batch_first=True)
        pad_y = pad_sequence(batch_y, batch_first=True)

        return pad_x, pad_y, len_x, len_y


class LSTM():
    def __init__(self, train_feature, train_label, valid_feature, valid_label, use_saved_model, train_end, valid_end, start_year):
        self.seq_len = 64
        self.train_end = train_end
        self.valid_end = valid_end
        self.start_year = start_year
        # self.period = period
        self.epoch = 50
        # if not use_saved_model:
        #     self.epoch = 100
        
        # def train_collate_fn(batch):
        #     batch_x = []
        #     batch_y = []
        #     len_x = []
        #     len_y = []
        #     for b in range(len(batch)):
        #         x, y = batch[b]
        #         p = np.random.random()
        #         length = 64
        #         start_index = 0
        #         if use_saved_model:
        #             if p > 0.3:
        #                 start_index = len(x) - length - 1#max(period * 6, 64)
        #                 start_index = max(0, start_index)
        #                 # start_index = min(len(x) - length - 1, start_index)
        #         # # print(len(x))
        #         # print(start_index, len(x) - length)
        #         index = np.random.randint(start_index, len(x) - length)
        #         batch_x.append(x[index: index + length])
        #         batch_y.append(y[index: index + length])
        #         # len_x.append(length)
        #         # len_y.append(length)

        #     len_x = torch.LongTensor([len(x) for x in batch_x])
        #     len_y = torch.LongTensor([len(y) for y in batch_y])

        #     pad_x = pad_sequence(batch_x, batch_first=True)
        #     pad_y = pad_sequence(batch_y, batch_first=True)
        #     return pad_x, pad_y, len_x, len_y

        train_data = TrainDataset(train_feature, train_label, self.seq_len)
        train_args = dict(shuffle=True, batch_size=512, num_workers=8, collate_fn=TrainDataset.collate_fn)
        # train_data = Weighted_TrainDataset(train_feature, train_label, self.seq_len)
        # train_args = dict(shuffle=True, batch_size=512, num_workers=8, collate_fn=train_collate_fn)
        self.train_loader = DataLoader(train_data, **train_args)

        val_data = TestDataset(valid_feature, valid_label, self.seq_len)
        val_args = dict(shuffle=False, batch_size=512, num_workers=8, collate_fn=TestDataset.collate_fn)
        self.val_loader = DataLoader(val_data, **val_args)

        test_data = TestDataset(train_feature, train_label, self.seq_len)
        test_args = dict(shuffle=False, batch_size=512, num_workers=8, collate_fn=TestDataset.collate_fn)
        self.test_loader = DataLoader(test_data, **test_args)
        
        self.input_dim = train_feature[0].shape[1]
        self.hidden_dim = 128
        self.output_dim = 1
        self.num_layers = 2
        self.model = LSTMModel(self.input_dim, self.hidden_dim, self.num_layers, self.output_dim).to(device)
        self.criterion = nn.L1Loss().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-6)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=5)
        print(use_saved_model)
        print("\n")
        if use_saved_model:
            checkpoint = torch.load("model.pth")
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def train(self):
        for epoch in range(self.epoch):
            print('Epoch', epoch + 1)
            training_loss = self.train_epoch()
            # val_loss, r2 = self.evaluate()


            self.scheduler.step()
            print("Epoch: "+str(epoch+1)+", Training loss: "+str(training_loss))#+", Validation loss:"+str(val_loss))
            if epoch != 0 and (epoch + 1) % 10 == 0:
                reals, predictions = self.plot(epoch + 1)   

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict()
        }, 'model.pth') 
          
        return reals, predictions

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        predictions = []
        reals = []
        for i, (inputs, targets, inputs_len, targets_len) in enumerate(self.train_loader):
            B = inputs.shape[0]
            inputs, targets= inputs.to(device), targets.to(device),
            self.optimizer.zero_grad()

            targets = torch.unsqueeze(targets, 2)
            out, lengths = self.model(inputs, inputs_len)
            # print(out.shape)
            # print(targets.shape)
            # print(inputs_len)

            for b in range(B):
                cut = inputs_len[b]
                out[b, cut:] -= out[b, cut:]
            # loss = self.criterion(out, targets)
            loss = self.loss_fn(out, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            
            # for b in range(B):
            #     pred = list(out[b, :inputs_len[b]].squeeze(1).detach().cpu().numpy().reshape(-1))
            #     real = list(targets[b, :inputs_len[b]].squeeze(1).detach().cpu().numpy().reshape(-1))
            #     predictions.extend(pred)
            #     reals.extend(real)
            # out = out.squeeze(2).detach().cpu().numpy().reshape((-1))
            # targets = targets.squeeze(2).detach().cpu().numpy().reshape((-1))
            # predictions.extend(list(out))
            # reals.extend(list(targets))

        # r2 = r2_score(reals, predictions)
        # print("train r2:" + str(r2))
        train_loss /= len(self.train_loader)
        return train_loss#, r2
    
    def predict(self):
        self.model.eval()
        predictions = []
        reals = []
        val_len = []
        val_loss = 0
        sum = 0
        print("validation")
        for i, (inputs, targets, inputs_len, targets_len) in enumerate(self.val_loader):
            B = inputs.shape[0]
            inputs, targets = inputs.to(device), targets.to(device)
            targets = torch.unsqueeze(targets, 2)
            
            out, lengths = self.model(inputs, inputs_len)
            loss = self.criterion(out, targets)
            val_loss += loss.item()

            for b in range(B):
                pred = list(out[b, inputs_len[b] - 1].detach().cpu().numpy().reshape(-1))
                real = list(targets[b, inputs_len[b] - 1].detach().cpu().numpy().reshape(-1))
  
                predictions.extend(pred)
                reals.extend(real)
            sum += B

        # val_loss /= len(self.val_loader)
        # r2 = r2_score(reals, predictions)
        # print("val_r2:" + str(r2))
        # print("val_loss:" + str(val_loss))
        # ax = plt.subplot(1, 2, 2)
        # plot_scatter_ci(ax, np.array(predictions), np.array(reals))
        # ax.set_title(f"Test Return v.s. Pred")     

        # plt.tight_layout()
        # plt.savefig("20220215/" + str(self.start_year + 3) + "/" + str(self.train_end) + "-" + str(self.valid_end) + "_" + str(epoch) + ".jpg")
        # plt.close()

        return reals, predictions

    def loss_fn(self, pred, label):
        dif = pred - label
        mse = torch.mean(dif ** 2)
        mul = pred * label
        mask = mul.lt(0.0)
        dir = torch.sum(mask) / len(mask)
        return mse + 0.1 * dir
    
    def plot(self, epoch):
        self.model.eval()
        fig = plt.figure(figsize=(30, 7))
         

        predictions = []
        reals = []
        train_loss = 0
        train_len = []
        print("train")
        for i, (inputs, targets, inputs_len, targets_len) in enumerate(self.test_loader):
            B = inputs.shape[0]
            inputs, targets = inputs.to(device), targets.to(device)
            targets = torch.unsqueeze(targets, 2)
            train_len.extend(inputs_len)
            out, lengths = self.model(inputs, inputs_len)
            loss = self.criterion(out, targets)
            train_loss += loss.item()


            for b in range(B):
                pred = list(out[b, :inputs_len[b]].squeeze(1).detach().cpu().numpy().reshape(-1))
                real = list(targets[b, :inputs_len[b]].squeeze(1).detach().cpu().numpy().reshape(-1))
                predictions.extend(pred)
                reals.extend(real)

        train_loss /= len(self.train_loader)
        r2 = r2_score(reals, predictions)
        print("train_r2:" + str(r2))
        print("train_loss:" + str(train_loss))
        ax = plt.subplot(1, 2, 1)  
        plot_scatter_ci(ax, np.array(predictions), np.array(reals))
        ax.set_title(f"Train Return v.s. Pred")
        
        # print("length:" + str(len(train_len)))
        predictions = []
        reals = []
        val_len = []
        val_loss = 0
        sum = 0
        print("validation")
        for i, (inputs, targets, inputs_len, targets_len) in enumerate(self.val_loader):
            B = inputs.shape[0]
            inputs, targets = inputs.to(device), targets.to(device)
            targets = torch.unsqueeze(targets, 2)
            
            out, lengths = self.model(inputs, inputs_len)
            loss = self.criterion(out, targets)
            val_loss += loss.item()

            for b in range(B):
                pred = list(out[b, inputs_len[b] - 1].detach().cpu().numpy().reshape(-1))
                real = list(targets[b, inputs_len[b] - 1].detach().cpu().numpy().reshape(-1))
  
                predictions.extend(pred)
                reals.extend(real)
            sum += B

        val_loss /= len(self.val_loader)
        r2 = r2_score(reals, predictions)
        print("val_r2:" + str(r2))
        print("val_loss:" + str(val_loss))
        ax = plt.subplot(1, 2, 2)
        plot_scatter_ci(ax, np.array(predictions), np.array(reals))
        ax.set_title(f"Test Return v.s. Pred")     

        plt.tight_layout()
        plt.savefig("20220314/" + str(self.start_year + 3) + "/" + str(self.train_end) + "-" + str(self.valid_end) + "_" + str(epoch) + ".jpg")
        plt.close()

        return reals, predictions


def plot_scatter_ci(ax, x, y, div_num=20):
    #sorted x
    ind = x.argsort()
    xmean = x.mean()

    sorted_x, sorted_y = x[ind], y[ind]
    xmin, xmax = sorted_x[0], sorted_x[-1]
    win_size = (xmax - xmin) / div_num
    step_size = win_size / 2

    cr_stds, cr_means, counts = [], [], []
    xs = []
    for i in range(2 * div_num - 1):
        mid = xmin + win_size / 2 + i * step_size
        start, end = mid - win_size / 2, mid + win_size / 2
        if end >= xmax: # out of border
            break
        ind_start = sorted_x.searchsorted(start)
        ind_end = sorted_x.searchsorted(end)
        xs.append(mid)
        counts.append(ind_end - ind_start)
        cr_means.append(sorted_y[ind_start:ind_end].mean())
        cr_stds.append(sorted_y[ind_start:ind_end].std())

    xs, counts = np.array(xs), np.array(counts)
    cr_means, cr_stds = np.array(cr_means), np.array(cr_stds)
    sqrt_counts = np.sqrt(counts)
    valid = sqrt_counts > 1
    xs, cr_means, cr_stds = xs[valid], cr_means[valid], cr_stds[valid]
    counts, sqrt_counts = counts[valid], sqrt_counts[valid]
    CW = 1.96 * cr_stds / sqrt_counts
    ax.plot(xs, cr_means, color="blue", alpha=0.8)
    ax.plot(xs, [xmean] * len(xs), 'r--', linewidth=1)
    ax.fill_between(xs, (cr_means - CW), (cr_means + CW),
        color='blue', alpha=0.2)
    mean, sigma = cr_means.mean(), cr_means.std()
    cr_means_ = cr_means[np.abs(cr_means - mean) < 3 * sigma]
    mean, sigma = cr_means_.mean(), cr_means_.std()
    try:
        ax.set_ylim([mean - 3 * sigma, mean + 3 * sigma])
    except:
        pass


def to_decile(real, prediction, boxplot):
    # deciles = [-8, -6, -4, -2, 0, 2, 4, 6, 8]
    # deciles = [-6, -2, 2, 6]
    length = len(deciles)
    reals = np.array(real)
    predictions = np.array(prediction)
    indice = np.argsort(prediction)
    predictions = predictions[indice]
    reals = reals[indice]

    backtest = reals[-50:]

    start = 0
    means = np.zeros(length + 1)
    medians = np.zeros(length + 1)
    stds = np.zeros(length + 1)
    for i in range(length):
        cut = deciles[i]
        end = np.searchsorted(predictions, cut, side='left') 
        pred = reals[start : end]

        boxplot[i].append(pred)
        means[i] = np.mean(pred)
        medians[i] = np.median(pred)
        stds[i] = np.std(pred)
        start = end

    pred = reals[end:]
    boxplot[-1].append(pred)
    means[-1] = np.mean(pred)
    medians[-1] = np.median(pred)
    stds[-1] = np.std(pred)
    
    return means, medians, stds, backtest


# def process_data(df, key_list, use_keylist, for_train):
#     df_groupby = df.groupby("code")
#     features = []
#     labels = []
#     length = []
#     code_list = []
#     l = df_groupby.groups.keys()
#     if use_keylist:
#         l = key_list
#     for key in l:
#         df_code = df_groupby.get_group(key)
#         if for_train and len(df_code) < 100:
#             continue
#         length.append(len(df_code))
#         if for_train:
#             code_list.append(key)
#         label = df_code['label'].to_numpy()
#         labels.append(label)
#         df_code = df_code.drop(columns=['code', 'date', 'label'])#, 'class'
#         feature = df_code.to_numpy()
#         features.append(feature)
#     return features, labels, code_list 


def process_data(df):
    df_groupby = df.groupby("code")
    features = []
    labels = []
    # length = []
    # code_list = []
    # l = df_groupby.groups.keys()
    # if use_keylist:
    #     l = key_list
    for key in df_groupby.groups.keys():
        df_code = df_groupby.get_group(key)
        # length.append(len(df_code))
        label = df_code['label'].to_numpy()
        labels.append(label)
        df_code = df_code.drop(columns=['code', 'date', 'label'])#, 'class'
        feature = df_code.to_numpy()
        features.append(feature)
    return features, labels#, code_list 


def main(): 
    df = pd.read_csv("./Data_Process/processed_with_turnover.csv")
    df = df[abs(df['adjclose']) < 0.11]
    df.drop(columns=['tradestatuscode', 'change', 'adjfactor', 'preclose', 'open', 'high', 'low', 'close', 
    'S_VAL_MV', 'S_DQ_MV', 'S_VAL_PE', 'S_VAL_PB_NEW', 'S_VAL_PCF_OCF', 'S_DQ_TURN', 'S_DQ_FREETURNOVER'], inplace=True)#, 'S_DQ_TURN', 'S_DQ_FREETURNOVER'
    scale_list = ['adjpreclose', 'adjopen', 'adjhigh', 'adjlow', 'adjclose']
    for col in scale_list:
        df[col] = df[col] * 100
    df.fillna(0, inplace=True)
    start_year = [2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]
    count = 0
    initial_date = 20100101

    for i in range(len(start_year)):
        start = start_year[i]
        test_start = (start + 3) * 10000 + 101
        test_end = (start + 4) * 10000 + 101
        train_date = df[(df['date'] >= test_start) & (df['date'] <= test_end)]['date'].unique()
        train_date = list(train_date)
        train_date.sort()
        num_date = len(train_date)
        period = int(num_date / 12)
        print("period:" + str(period))
        
        # for j in range(len(period)):
            # start_date = start * 10000 + 101
        date = start + 3
        real = []
        prediction = []
        box_plots = [[] for w in range(len(deciles) + 1)]

        deciles_means = np.zeros((num_date, len(deciles) + 1))
        deciles_medians = np.zeros((num_date, len(deciles) + 1))
        deciles_stds = np.zeros((num_date, len(deciles) + 1))

        time = []
        
        df_train = df.loc[(df['date'] >= initial_date) & (df['date'] < train_date[0])]
        temp = df_train.copy()
        gb = temp.groupby('code').count().reset_index().loc[:, ['code', 'date']]
        gb = gb[gb['date'] >= 100].loc[:, ['code']]
        # df_train = df_train.merge(gb, left_on="code", right_on="code", how="right")
        df_roll = df.merge(gb, left_on="code", right_on="code", how="right")
        account = Backtest(topk=50, n_drop=10)#, time=df_train['date'].max())
        returns = []
        values = []
        pos_ratio = []
        neg_ratio = []

        clean = False
        restart = 5
        clean_count = 0
        for l in range(num_date - 1):
            # print("clean:" + str(clean))
            # if clean:
            #     clean_count += 1
            #     if clean_count == restart:
            #         clean = False
            #         clean_count = 0
            # train_end = date * 10000 + train_date[l]
            # valid_end = date * 10000 + train_date[l + period[j]]
            train_end = train_date[l]
            valid_end = train_date[l + 1]
            # cur_time = train_end

            df_train = df_roll.loc[(df_roll['date'] >= initial_date) & (df_roll['date'] < train_end)]
            df_valid = df_roll.loc[(df_roll['date'] >= initial_date) & (df_roll['date'] < valid_end)]

        #     train_feature, train_label, key_list = process_data(df_train, [], False, True)
        #     valid_feature, valid_label, key_list= process_data(df_valid, key_list, True, False)
            train_feature, train_label = process_data(df_train)
            valid_feature, valid_label = process_data(df_valid)
            
            if count == 0:
                lstm = LSTM(train_feature, train_label, valid_feature, valid_label, False, train_end, valid_end, start)#, period[j])
            else:
                lstm = LSTM(train_feature, train_label, valid_feature, valid_label, True, train_end, valid_end, start)#, period[j])
            count += 1

            if l % period == 0:
                reals, predictions = lstm.train()
                clean = False
            else:
                reals, predictions = lstm.predict()
            
            # print(len(reals))
            # print(len(predictions))
            code = df_valid['code'].unique()
            df_pred = pd.DataFrame(data={'code': code, 'pred': predictions, 'real': reals})
            # print(df_pred)
            

            for_valid = df_valid[df_valid['date'] == train_end]
            df_merge = for_valid.merge(df_pred, left_on="code", right_on="code", how="left").loc[:, ['code', 'pred', 'label']]
            total_len = len(df_merge)
            pos_len = len(df_merge[df_merge['pred'] > 0])
            neg_len = len(df_merge[df_merge['pred'] < 0])
            pos_ratio.append(pos_len / total_len)
            neg_ratio.append(neg_len / total_len)
            if not clean:
                ret, cum_amount = account.trade(df_merge, start + 3, valid_end)
            else:
                ret = 1
            returns.append(ret)
            values.append(cum_amount)
            # if len(values) >= 5:
            #     print("ratio:" + str(values[-1] / values[-5]))
            # if len(values) >= 5 and not clean and values[-1] / values[-5] <= 0.95 :
            #     clean = True
            #     trade_cost = account.clean()
            #     returns[-1] *= (1 - trade_cost)
            #     values[-1] *= (1 - trade_cost)
            #     cum_amount *= (1 - trade_cost)
            
        print(values[-1])
        x = np.arange(len(returns))

        # Set up a subplot grid that has height 2 and width 1,
        # and set the first such subplot as active.
        plt.subplot(3, 1, 1)

        # Make the first plot
        plt.plot(x, returns)
        plt.title('Daily Returns')
        plt.grid(visible=True, axis='y')

        # Set the second subplot as active, and make the second plot.
        plt.subplot(3, 1, 2)
        plt.plot(x, values)
        plt.title('Cumulative Position')
        plt.grid(visible=True, axis='y')

        plt.subplot(3, 1, 3)
        plt.plot(x, pos_ratio)
        plt.plot(x, neg_ratio)
        plt.title('Distribution')
        plt.legend(['Positive', 'Negative'])

        plt.tight_layout()

        # Show the figure.
        plt.savefig("20220314/" + str(start + 3) + "/backtest.jpg")
        plt.close()
        
        #     decile_means, decile_medians, decile_stds, backtest = to_decile(reals, predictions, box_plots)
        #     print(backtest)
        #     print(np.mean(backtest))
        #     deciles_means[count] = decile_means
        #     deciles_medians[count] = decile_medians
        #     deciles_stds[count] = decile_stds
            
        #     time.append(str(train_end) + "-" + str(valid_end))
        #     real.extend(reals)
        #     prediction.extend(predictions)
        #     count += 1

        # date += 1

        # fig = plt.figure(figsize=(30, 7))
        # ax = plt.subplot(1, 1, 1)  
        # plot_scatter_ci(ax, np.array(prediction), np.array(real))
        # ax.set_title(f"Test Return v.s. Pred")     
        # plt.tight_layout()
        # plt.savefig("20220215/" + str(start + 3) + "/final.jpg")
        # plt.close()

        # decile_means, decile_medians, decile_stds, backtest = to_decile(real, prediction, box_plots)
        # deciles_means[-1] = decile_means
        # deciles_medians[-1] = decile_medians
        # deciles_stds[-1] = decile_stds

        # time.append("total")
        # fig, ax = plt.subplots(len(deciles) + 1, 1, figsize=(50, 30))
        
        # deciles_means = deciles_means.T
        # deciles_medians = deciles_medians.T
        # deciles_stds = deciles_stds.T
        
        # for i in range(len(deciles) + 1):
        #     ax[i].plot(time, deciles_means[len(deciles) - i], 'bo-', label = 'mean')
        #     ax[i].plot(time, deciles_medians[len(deciles) - i], 'ro-', label = 'median')
        #     ax[i].plot(time, deciles_stds[len(deciles) - i], 'yo-', label = 'std')
        #     ax[i].set_title("class " + str(i + 1))
        #     ax[i].legend()
        
        # plt.savefig("20220215/" + str(start + 3)  + "/summary.jpg")
        # plt.close()

        # fig, ax = plt.subplots(len(deciles) + 1, 1, figsize=(30, 30))
        # for i in range(len(deciles) + 1):
        #     ax[i].boxplot(box_plots[len(deciles) - i])
        #     ax[i].set_title("class " + str(i + 1))
        #     # ax[i].legend()
        #     # ax[i].xticks(rotation=-90)
        
        # plt.savefig("20220215/" + str(start + 3) + "/boxplot.jpg")
        # plt.close()


if __name__ == "__main__":
    main()