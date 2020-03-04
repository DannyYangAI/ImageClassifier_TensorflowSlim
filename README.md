參考資料來源：https://blog.csdn.net/LiJiancheng0614/article/details/77727445  使用TensorFlow-Slim進行圖像分類: 作者lijiancheng0614 


步驟：
1.準備好圖片資料，依類別分類

2.開啟filesToList.py，修改圖片資料夾名稱與類別名稱後執行此py檔，將所有圖片路徑輸出成list.txt

3.執行 CreateTrainTestDataset.py, 讀取 list.txt ，會切分資料，輸出 list_train.txt list_val.txt，記得要在程式碼中設定測試筆數
[重要!] val的資料與train 過程沒有任何關系!!! 是用於訓練完之模組驗證用的。

4.下載 tensorflow model (https://github.com/tensorflow/models/tree/master/research/slim)，
放入D:\Project\ImageClassifier 資料夾命名models ，此步驟做過一次即可

5.執行 CreateTFRcord.py , 生成訓練與驗證TFRcord檔各5個

6. 可載入預訓練之model ckpt檔, 放到 D:\Project\ImageClassifier\models\research\slim    EX : http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
   
7. 於D:\Project\ImageClassifier\models\research\slim\datasets，參考flowers.py，建立 dataset_classification.py，用於讀入資料， 此步驟做過一次即可

8.修改 D:\Project\ImageClassifier\models\slim\train_image_classifier.py ， 參考 作者lijiancheng0614 建議，此步驟做過一次即可，目的是可以讀自己提供的資料集

9.執行：出現V1什麼錯誤的，把錯誤行的"compat.v1" 拿掉(但是太多拿不完)，mbilenet V2的錯誤可以拿掉 (XXXfactory.py) , 這些問題都是tensorflow版本的問題目前使用1.13.1

checkpoint 的路徑錯誤：https://www.cnblogs.com/weiyinfu/p/10071955.html, windows才會出現，可以直接放ckpt到執行目錄，就不會有錯誤

10. 訓練的 執行指令


    python train_image_classifier.py --train_dir=train_logs --dataset_dir=D:/Project/ImageClassifier/data/train --num_samples=3320 --num_classes=5 --labels_to_names_path=D:/Project/ImageClassifier/data/labels.txt --model_name=inception_resnet_v2 --checkpoint_path=inception_resnet_v2_2016_08_30.ckpt --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits

11.輸出結果：
    訓練得到的模型存放路徑 D:\Project\ImageClassifier\models\research\slim\train_logs
    



