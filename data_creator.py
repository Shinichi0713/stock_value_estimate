# モジュールのインポート
# 時系列データ取得
from pandas_datareader import data as wb
import yfinance as yfin
import torch
import numpy as np
import common_resource

# yahooファイナンスから株価の時系列データを取得
def download_dataset():
    yfin.pdr_override()

    # 株価コード
    # 株式分割のない銘柄を選択
    stock_codes = [
        # '6367.T', # ダイキン工業
        # '8001.T', # 伊藤忠商事
        # '6501.T', # トヨタ自動車
        '8725.T'    # ＭＳ＆ＡＤ
    ]

    # 取得開始日
    start_date='2003-1-1'
    # 取得終了日
    end_date='2022-12-31'

    # データ読出し
    dataframes = {}
    for stock_code in stock_codes:
        df=wb.DataReader(stock_code,start=start_date,end=end_date)
        dataframes[stock_code] = df

    return dataframes


def show_data(dataframes):
    from matplotlib import pyplot as plt
    for key, df in dataframes.items():
        title = key
        plt.figure()
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        df = df['Close']
        time = df.index.to_numpy()
        data = df.values
        plt.plot(time, data, color='black',
                linestyle='-', label='close')
        # plt.plot(time, df['25MA'], color='red',
        #         linestyle='--', label='25MA')
        plt.legend()  # 凡例
        plt.savefig(f'{title}.png')  # 図の保存
        plt.show()

def create_dataset():
    # データ取得
    dataframes = download_dataset()

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    # 時系列データ
    datasets_train = {}
    datasets_eval = {}
    for key, df in dataframes.items():
        # 正規化
        mean_list=df.mean().values
        std_list=df.std().values
        df=(df-mean_list)/std_list
        inout_data = []
        for i in range(len(df)-common_resource.observation_period_num-common_resource.predict_period_num):
            data=df.iloc[i:i+common_resource.observation_period_num,4].values
            label=df.iloc[i+common_resource.predict_period_num:i+common_resource.observation_period_num+common_resource.predict_period_num,4].values
            inout_data.append((data,label))
        inout_data=torch.FloatTensor(inout_data)

        dataset_train=inout_data[:int(np.shape(inout_data)[0]*common_resource.train_rate)].to(device)
        dataset_eval=inout_data[int(np.shape(inout_data)[0]*common_resource.train_rate):].to(device)
        datasets_train[key] = dataset_train
        datasets_eval[key] = dataset_eval

        print('train data：',np.shape(dataset_train)[0])
        print('valid data：',np.shape(dataset_eval)[0])
    
    return datasets_train, datasets_eval

if __name__ == '__main__':
    # create_dataset()

    dataframes = download_dataset()
    show_data(dataframes)



