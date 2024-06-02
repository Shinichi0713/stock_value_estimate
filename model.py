

import torch
import torch.nn as nn
import numpy as np
import os, sys
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt
import common_resource, data_creator

dir_current = os.path.dirname(__file__)

class StockPricePredictor(nn.Module):
    def __init__(self, input_size=common_resource.observation_period_num*common_resource.num_explain, hidden_layer_size=common_resource.hidden_size, output_size=common_resource.num_explain, num_layers=4):
        super(StockPricePredictor, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_layer_size
        # LSTMレイヤー
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        # 線形レイヤー
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        # デバイス
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # NNパラメータのロード
        if os.path.exists(dir_current + '/model.pth'):
            print('load model parameters.')
            self.load_state_dict(torch.load(dir_current + '/model.pth'))
        self.to(self.device)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1))
        # h0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_dim).to(self.device)
        # c0 = torch.zeros(self.num_layers, input_seq.size(0), self.hidden_dim).to(self.device)
        
        # lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1), (h0, c0))
        lstm_out = self.leaky_relu(lstm_out)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return torch.sigmoid(predictions)

# データを分割してデータ化
def get_batch(source, i, batch_size):
    seq_len=min(batch_size, len(source)-1-i)
    data=source[i:i+seq_len]
    input=data[:, :common_resource.observation_period_num*common_resource.num_explain]
    target=data[:, common_resource.observation_period_num*common_resource.num_explain:]

    return input, target

def train_model():
    dir_current = os.path.dirname(__file__)

    datasets_train, datasets_eval = data_creator.create_dataset()
    model = StockPricePredictor()
    keys_stock = datasets_train.keys()
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=common_resource.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1.0,gamma=0.95)
    num_epochs = 50

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
            model.cpu()
            torch.save(model.state_dict(), dir_current + '/model.pth')
            model.to(model.device)
        print(f'epoch: {epoch}, loss: {total_loss_eval}')

        if total_loss_eval < 0.001:
            break

def evaluate_model():
    dir_current = os.path.dirname(__file__)

    datasets_eval = data_creator.create_data_evaluation()
    model = StockPricePredictor()
    keys_stock = datasets_eval.keys()

    model.eval()
    data_eval = []
    data_real = []
    for key_stock in keys_stock:
        with torch.no_grad():
            dataset_eval = datasets_eval[key_stock]
            x, _ = get_batch(dataset_eval, 0, 1)
            for batch, i in enumerate(range(0,len(dataset_eval) - common_resource.observation_period_num,1)):
                _, y = get_batch(dataset_eval, i, 1)
                x = x.to(model.device)  # 説明変数
                y = y.to(model.device)  # 予測値
                output = model(x)
                data_eval.append(output[0].cpu().numpy())
                data_real.append(y[0].cpu().numpy())
                # 次の説明変数を作成
                x = torch.cat([x[0, common_resource.num_explain:], output[0]]).view(1, -1)
        data_eval = np.array(data_eval)
        data_real = np.array(data_real)
        # 画像に表示
        
        fig = plt.figure()
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.set_title("close value", fontsize=20)
        ax1.plot(data_eval[:, 3], label='predict')
        ax1.plot(data_real[:, 3], label='real')
        ax1.legend()
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.set_title("volume", fontsize=20)
        ax2.plot(data_eval[:, 6], label='predict')
        ax2.plot(data_real[:, 6], label='real')
        ax2.legend()
        plt.savefig(dir_current + f'/stock_{key_stock}.png')
        plt.show()


if __name__ == '__main__':
    train_model()
    evaluate_model()