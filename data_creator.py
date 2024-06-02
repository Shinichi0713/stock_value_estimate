# モジュールのインポート
# 時系列データ取得
from pandas_datareader import data as wb
import yfinance as yfin
import torch
import numpy as np
from sklearn.model_selection import train_test_split
import common_resource
import pandas as pd

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
    start_date='2008-1-1'
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
        in_data = []
        out_data = []
        for i in range(len(df)-common_resource.observation_period_num-common_resource.predict_period_num):
            for j in [0,1,2,3,5]:
                data=df.iloc[i:i+common_resource.observation_period_num,j].values
                in_data.append(data)
                label=df.iloc[i+common_resource.observation_period_num:i+common_resource.observation_period_num+common_resource.predict_period_num,j].values
                out_data.append(label)
        in_data=np.array(in_data).reshape(-1,common_resource.observation_period_num*common_resource.num_explain)
        out_data=np.array(out_data).reshape(-1,common_resource.predict_period_num*common_resource.num_explain)
        inout_data=np.concatenate((in_data,out_data),axis=1)
        dataset_train, dataset_eval = train_test_split(inout_data, test_size=0.2)
        dataset_train=torch.FloatTensor(dataset_train)
        dataset_eval=torch.FloatTensor(dataset_eval)

        datasets_train[key] = dataset_train
        datasets_eval[key] = dataset_eval

        print('train data：',np.shape(dataset_train)[0])
        print('valid data：',np.shape(dataset_eval)[0])
    
    return datasets_train, datasets_eval

def create_data_evaluation():
    # データ取得
    dataframes = download_dataset()
    # 時系列データ
    datasets_output = {}

    index_start = 100
    for key, df in dataframes.items():
        # 正規化
        mean_list=df.mean().values
        std_list=df.std().values
        df=(df-mean_list)/std_list
        inout_data = []
        in_data = []
        out_data = []
        for i in range(len(df)-common_resource.observation_period_num-common_resource.predict_period_num-index_start, len(df)-common_resource.observation_period_num-common_resource.predict_period_num-index_start+100):
            for j in [0,1,2,3,5]:
                data=df.iloc[i:i+common_resource.observation_period_num,j].values
                in_data.append(data)
                label=df.iloc[i+common_resource.observation_period_num:i+common_resource.observation_period_num+common_resource.predict_period_num,j].values
                out_data.append(label)
        in_data=np.array(in_data).reshape(-1,common_resource.observation_period_num*common_resource.num_explain)
        out_data=np.array(out_data).reshape(-1,common_resource.predict_period_num*common_resource.num_explain)
        inout_data=np.concatenate((in_data,out_data),axis=1)
        inout_data=torch.FloatTensor(inout_data)
        datasets_output[key] = inout_data
        print('train data：',np.shape(inout_data)[0])
    
    return datasets_output

def create_predict_dataset():
    import os
    dir_current = os.path.dirname(__file__)
    path_target = dir_current + '/' + common_resource.path_predict
    # データ取得
    df = pd.read_csv(path_target, header=0, index_col=0)
    df["Open"] = (df["Open"].str.replace(",", "").astype(float))[::-1].values
    df["High"] = (df["High"].str.replace(",", "").astype(float))[::-1].values
    df["Low"] = (df["Low"].str.replace(",", "").astype(float))[::-1].values
    df["Close"] = (df["Close"].str.replace(",", "").astype(float))[::-1].values
    df["Volume"] = (df["Volume"].str.replace(",", "").astype(float))[::-1].values
    mean_list=df.mean().values
    std_list=df.std().values
    dataframes=(df-mean_list)/std_list
    in_data = []

    for j in [0,1,2,3,5]:
        data=dataframes.iloc[0:0+common_resource.observation_period_num,j].values
        in_data.append(data)

    in_data=np.array(in_data).reshape(-1,common_resource.observation_period_num*common_resource.num_explain)

    in_data=torch.FloatTensor(in_data)
    return in_data

if __name__ == '__main__':
    # create_data_evaluation()

    # dataframes = download_dataset()
    # show_data(dataframes)

    print(create_predict_dataset())



