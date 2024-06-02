

import torch
import torch.nn as nn
import numpy as np
import os, sys
from torch.utils.data import TensorDataset, DataLoader
import common_resource, data_creator

class StockPricePredictor(nn.Module):
    def __init__(self, input_size=common_resource.observation_period_num*5, hidden_layer_size=common_resource.hidden_size, output_size=5, num_layers=2):
        super(StockPricePredictor, self).__init__()

        # LSTMレイヤー
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)

        # 線形レイヤー
        self.linear = nn.Linear(hidden_layer_size, output_size)

        # デバイス
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions

# データを分割してデータ化
def get_batch(source, i, batch_size):
    seq_len=min(batch_size, len(source)-1-i)
    data=source[i:i+seq_len]
    input=data[:, :common_resource.observation_period_num*5]
    target=data[:, common_resource.observation_period_num*5:]

    return input, target

def train_model():
    dir_current = os.path.dirname(__file__)

    datasets_train, datasets_eval = data_creator.create_dataset()
    model = StockPricePredictor()
    keys_stock = datasets_train.keys()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=common_resource.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1.0,gamma=0.95)
    num_epochs = 100

    total_loss_eval_best=sys.float_info.max
    for epoch in range(num_epochs):
        model.train()
        total_loss_train=0.0
        for key_stock in keys_stock:
            dataset_train = datasets_train[key_stock]
            for batch, i in enumerate(range(0,len(dataset_train),common_resource.batch_size)):
                x, y = get_batch(dataset_train, i, common_resource.batch_size)
                x = x.to(model.device)
                y = y.to(model.device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                total_loss_train+=loss.item()
        scheduler.step()

        model.eval()
        total_loss_eval=0.0
        with torch.no_grad():
            dataset_eval = datasets_eval[key_stock]
            for batch, i in enumerate(range(0,len(dataset_eval),common_resource.batch_size)):
                x, y = get_batch(dataset_eval, i, common_resource.batch_size)
                x = x.to(model.device)
                y = y.to(model.device)
                output = model(x)
                loss = criterion(output, y)
                total_loss_eval+=loss.item()
        # モデルの保存
        if total_loss_eval_best>total_loss_eval:
            total_loss_eval_best=total_loss_eval
            torch.save(model.state_dict(), dir_current + '/model.pth')
        print(f'epoch: {epoch}, loss: {total_loss_eval}')

        if total_loss_eval < 0.001:
            break


if __name__ == '__main__':
    train_model()