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
初めのデータのみ実データで、以降のデータは実データを削除→モデルの推定値で置き換えとして予測を行った。<br>
データは3か月分を予測させたが初日の動向程度であれば予測できそうであることが分かった。<br>
他方、長期のトレンドはいずれも上昇というように予測されているが大外れをするケースがある。<br>
また、長期になればなるほど、脈動が小さくなっていくことが確認された。<br>
→LSTMによる長期の予測は、トレンドは予測できるかもしれないが、脈動や山の予測は難しい<br>
![image](https://github.com/Shinichi0713/stock_value_estimate/assets/61480734/85b02fc3-cca6-4599-b1a1-fe7ac9e76a8b)

![image](https://github.com/Shinichi0713/stock_value_estimate/assets/61480734/deada9f7-26f6-4cf9-8a94-7a04fac107a0)

![image](https://github.com/Shinichi0713/stock_value_estimate/assets/61480734/3d4ad251-507c-49a7-bd48-238a4b0ce990)
