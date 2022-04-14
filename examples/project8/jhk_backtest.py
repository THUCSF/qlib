import numpy as np
import pandas as pd

class Backtest():
    def __init__(self, topk=20, n_drop=5) -> None:
        self.position = Position(topk)
        self.topk = topk
        self.n_drop = n_drop
        self.cum_amount = 1

    def trade(self, df, year, date, clean):
        if clean:
            sell_trade_cost = 0
            if self.position.hold is not None:
                order = Order(0, 0, "sell", None)
                sell_trade_cost = self.position.trade(order)
            return 1 - sell_trade_cost, self.cum_amount * (1 - sell_trade_cost)
        df.sort_values(by=['pred'], ascending=False, inplace=True)
        selected = df.iloc[:self.topk]
        print("topk")
        print(selected)
        
        # 修改路径
        selected.to_csv("./result/20220331/" + str(year) + "/" + "top_k" + ".csv" , index=False, header=True, mode="a")
        if self.position.hold is None:
            # 修改路径
            selected.to_csv("./result/20220331/" + str(year) + "/" + "position" + ".csv" , index=False, header=True, mode="a")
            order = Order(0, 0, "buy", selected)
            trade_cost = self.position.trade(order)

            hold = self.position.hold.copy()
            stocks = selected['label'].to_numpy() / 100
            ret = np.sum(1 + stocks) / self.topk - trade_cost
            print(np.mean(stocks))
            self.cum_amount *= ret
        else:
            stocks = selected['label']
            hold = self.position.hold.copy()
            hold = hold.merge(df, left_on="code", right_on="code", how="inner")

            hold.sort_values(by=['pred'], ascending=False, inplace=True)
            sell = hold[hold['pred'] < 0]
            if len(sell) < self.n_drop:
                sell = hold.iloc[-self.n_drop:]
            order = Order(0, 0, "sell", sell)
            sell_trade_cost = self.position.trade(order)

            buy_length = self.topk - len(self.position.hold)
            buy_candidate = list(set(selected['code'].to_numpy()) - set(self.position.hold['code'].to_numpy()))
            df_buy = pd.DataFrame(data={"code": buy_candidate})
            df_buy = df_buy.merge(selected, left_on="code", right_on="code", how="inner")
            df_buy.sort_values(by=['pred'], ascending=False, inplace=True)
            buy = df_buy.iloc[:buy_length]
            order = Order(0, 0, "buy", buy)
            buy_trade_cost = self.position.trade(order)
            
            stocks = self.position.hold.merge(df, on="code", how="left")
            stocks.fillna(0, inplace=True)
                
            print("current hold")
            print(stocks)
            # 修改路径
            stocks.to_csv("./result/20220331/" + str(year) + "/" + "position" + ".csv", index=False, header=True, mode="a")

            stocks = stocks['label'].to_numpy() / 100
            
            print("mean return:" + str(np.mean(stocks)))
            print("length:" + str(len(stocks)))
            
            ret = np.sum(1 + stocks) / self.topk - buy_trade_cost - sell_trade_cost# - morning_trade_cost
            self.cum_amount *= ret
        
        print("return:" + str(ret))
        print("cum:" + str(self.cum_amount))
        return ret, self.cum_amount
    
    def clean(self):
        trade_cost = self.position.trade(Order(0, 0, "sell", None))
        return trade_cost
        
class Position():
    def __init__(self, topk) -> None:
        self.hold = None
        self.topk = topk
    
    def trade(self, order):
        if order.dir == "buy":
            trade_cost = self.buy(order)
        else:
            trade_cost = self.sell(order)
        
        return trade_cost

    def buy(self, order):
        if self.hold is None:
            self.hold = order.stocks.loc[:, ['code']]
        else:
            self.hold = pd.concat([self.hold, order.stocks.loc[:, ['code']]])
        return (1.5 / 1000) * len(order.stocks) / self.topk
    
    def sell(self, order):
        if order.stocks is None:
            self.hold = None
            return 2.5 / 1000

        self.hold = self.hold.merge(order.stocks, on="code", how="outer")
        self.hold = self.hold[self.hold['pred'].isna()]
        self.hold = self.hold.loc[:, ['code']]
        return (2.5 / 1000) * len(order.stocks) / self.topk
        


class Order():
    def __init__(self, cur_time, order_time, dir, stocks) -> None:
        self.cur_time = cur_time
        self.order_time = order_time
        self.dir = dir
        self.stocks = stocks


