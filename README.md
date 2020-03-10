
使用tf-slim 做影像辨識的優點 , 強化功能如下

1 不需重複寫code, 可任意更換下列神經網路訓練
     
     Inception V1-V4
     Inception-ResNet-v2
     ResNet V1 50 -V2 200(層)
     VGG 16 19
     MobileNet_v1_1.0_224 v2_1.0_224
     NASNet-A_Mobile and Large
     PNASNet-5_Mobile and Large

2.提供google預訓練模型(牽移學習)直接使用

3.中斷訓練時得到的模型可接續訓練(牽移學習)

4.訓練程式、驗證程式、推論程式，全部可以使用多GPU執行，下參數即可使用

5.使用tf-record較有效率的輸入訓練資料格式檔

6.不同網路有專屬的影像前處理程式，減少資料輸入格式錯誤問題

7.完成驗證程式：可讀取訓練好的模型與驗證用的TFRecord檔，分析後輸出FN、FP、TN、TP、Recall、Precision、Accuracy及混沌矩陣(多GPU執行)

8. 完成推論程式，讀取圖檔資料夾即可推論全部(未來優化：初始化一次即可)

9. 與物件偵測相同tensorflow 版本執行(1.13.1)


####============================================================================================




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
    
    python train_image_classifier.py --batch_size=512 --train_dir=train_logs --dataset_dir=D:/Project/ImageClassifier/data/train --num_samples=3320 --num_classes=5 --labels_to_names_path=D:/Project/ImageClassifier/data/labels.txt --model_name=inception_resnet_v2 --checkpoint_path=inception_resnet_v2_2016_08_30.ckpt --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits

11.輸出結果：
    訓練得到的模型存放路徑 D:\Project\ImageClassifier\models\research\slim\train_logs

12.tensorboard：D:\Project\ImageClassifier\models\research\slim>tensorboard --logdir=train_logs/ --host localhost --port 8088

瀏覽器：http://localhost:8088 

13.驗證：修改 D:\Project\ImageClassifier\models\slim\eval_image_classifier.py ， 參考 作者lijiancheng0614 建議，此步驟做過一次即可，目的是可以讀自己提供的資料集

14驗證的 執行指令

python train_image_classifier.py --batch_size=512 --train_dir=train_logs --dataset_dir=D:/Project/ImageClassifier/data/train --num_samples=3320 --num_classes=5 --labels_to_names_path=D:/Project/ImageClassifier/data/labels.txt --model_name=inception_resnet_v2 --checkpoint_path=inception_resnet_v2_2016_08_30.ckpt --checkpoint_exclude_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits --trainable_scopes=InceptionResnetV2/Logits,InceptionResnetV2/AuxLogits

    
"""""""""""
以下為部分參數說明：

--train_dir表示訓練權重，tensorboard等信息保存的地址

--dataset_dir表示轉出來的tfrecords保存的地址

--dataset_name表示我們現在訓練哪個數據集，因爲不同的數據集，在解析方式上是有些微差別的

--dataset_split_name表示我們現在是用pascalvoc_2007數據集的train數據還是test數據

--model_name表示我們模型的名字，檢測模型是SSD，輸入是300x300，base network是VGG

--checkpoint_path就是VGG pre-trained weights存放的地址

--checkpoint_exclude_scopes表示這些參數不需要從checkpoint中恢復

--trainable_scopes表示訓練的時候只更新這些參數的值

decay :直观解释：假设给定初始学习率learning_rate为0.1，学习率衰减率为0.1，decay_steps为10000。
则随着迭代次数从1到10000，当前的学习率decayed_learning_rate慢慢的从0.1降低为0.10.1=0.01，
当迭代次数到20000，当前的学习率慢慢的从0.01降低为0.10.1^2=0.001，以此类推。 也就是说每10000次迭代，学习率衰减为前10000次的十分之一，该衰减是连续的，这是在staircase为False的情况下。如果staircase为True，则global_step / decay_steps始终取整数，也就是说衰减是突变的，每decay_steps次变化一次，变化曲线是阶梯状。

      num_clones : 每個單機部署多少個clone（即部署在多少個GPU）
      clone_on_cpu : 如果為True，則單機中的每個clone將被放在CPU中
      replica_id :   整數，模型所部署的單機的索引，通常是0.  
      num_replicas: 使用多少個單機，通常為1，表示單機部署。此時`worker_device`, `num_ps_tasks`和 `ps_device`這幾個引數將被忽略。
      num_ps_tasks: ‘ps’作業(分散式作業)使用多少個單機，如果為0表示不使用單機
      worker_job_name: 預設為“worker”
      ps_job_name:預設為'ps'
"""""""""""""""""""""

logdir指定檢查點和事件文件的存儲目錄。

number_of_steps我們可以限制採取任何數字的梯度步數。比如要求上限1000個步驟。

save_summaries_secs = 300表示將每隔5分鐘計算摘要

save_interval_secs = 600表示將每10分鐘保存一次模型檢查點


