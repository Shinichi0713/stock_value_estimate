# stock_value_estimate
# やりたいこと
LSTMにより株価予測モデルを作り、少なくとも3営業日先程度のスイングトレードに活用できるようにする

参考論文
https://reader.elsevier.com/reader/sd/pii/S1877050920304865?token=5FDF3E3A4697187DEFC3F0EFE0C2F6C8844529F3D5643352620F7EBB7ECF685926824BCE52BB7BC311F2C4511CAF5729&originRegion=us-east-1&originCreation=20220131034120


# ターゲット銘柄
MS&AD社の株価をターゲットとする
![image](https://github.com/Shinichi0713/stock_value_estimate/assets/61480734/b842a2bd-12bf-4728-a117-1ff155a63d1c)


400, 25, 2, 0.15207472257316113<br>
600, 25, 3, 0.11646148934960365<br>
1000, 25, 4, 0.09665982890874147<br>
隠れ層, データ数, レイヤ数


# 学習後の推定結果

![image](https://github.com/Shinichi0713/stock_value_estimate/assets/61480734/85b02fc3-cca6-4599-b1a1-fe7ac9e76a8b)

