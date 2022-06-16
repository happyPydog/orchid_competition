# Orchid Species Identification and Classification Contest

## [Colab version](https://colab.research.google.com/drive/11_yGGLCv-MORehj59de9q-O1Skx4dZBc)

-----------------------------------------------------------------

# 環境 `python == 3.9.5` 

以下是執行整個程式的流程。
## 1. 下載整個 repository
    $git clone https://github.com/Rammstein-1994/orchid_competition.git)

## 2. `cd` 到 `orchid_competition` 資料夾，並從 `requirements.txt` 下載套件
`$cd orchid_competition`

`$pip install -r requirements.txt`

## 3. 下載 `training.zip` 並放到 `orchid_competition` 資料夾中
[training.zip](https://drive.google.com/file/d/1KT_mJEdYtOXF79gdwgQsjmZQfzQS3ApU/view?usp=sharing) (如果電腦裡面已經有training.zi，省略下載的步驟)

以下是目前資料夾的結構圖
```bash
   orchid_competition
   ├── src
   ├── .gitignore
   ├── ...
   ├── ...
   ├── training.zip
```
## 4. 執行 `prepare_data.py`

執行 prepare_data.py 主要是將整個訓練資料拆分成 train 和 test，預設 test size 為 0.2，random seed 設為 22，這些參數都可以自由地更改。

    $python prepare_data.py --test_size 0.2 --img_dir "training" --csv_dir "trianing/label.csv" --save_dir "orchid_competition" --random_state 22

執行完之後會多出 orchid_dataset 和 training 兩個資料夾

```bash
   orchid_competition
   ├── orchid_dataset
   │   ├── test 
   │   │   ├── 0
   │   │   ├── 1
   │   │   ...
   │   │ 
   │   ├── train
   │   │   ├── 0
   │   │   ├── 1
   │   │   ...
   │   │
   │   ├── test.csv
   │   ├── train.csv
   │ 
   ├── training
   │   ├── 0a1h7votc5.jpg
   │   ...
   ...
```
接下來就可以開始訓練模型了
## 5. 訓練模型，執行 `swinv2_transformer_training_192.py`

我們會先在 192x192 解析度上進行訓練，等模型訓練到 200 個 epochs 之後就將模型存起來，接下來就會用 384x384 解析度進行微調。

    $python swinv2_transformer_training_192.py --config ...

因為 swinv2_transformer_training_192.py 參數很多，如果不想自己設定就直接執行就執行以下，會用我們在告報中使用的參數設定

     $python swinv2_transformer_training_192.py --BATCH_SIZE 16

如果 GPU 記憶體不夠就把 BATCH_SIZE 調小一點

執行結束後會產生 `swinv2_base_window12_192_22k.pt` 的模型存檔

## 6. Fine-tune on high resolution，執行 `swinv2_transformer_fine_tune_384.py`

在上一步驟我們已經訓練好 swinv2_transformer_training_192 模型，並將模型存在 `swinv2_base_window12_192_22k.pt` 中，在 fine tune 時我們要把它 load 到 fine-tune 使用的模型中

`--CHECKPOINT` 是用來讀取在192解析度上訓練好的存檔路徑

`--CHECKPOINT <swinv2_base_window12_192_22k path>`

    $python swinv2_transformer_fine_tune_384.py --CHECKPOINT "swinv2_base_window12_192_22k.pt"

最後訓練完這個程式後，就會輸出 `swinv2_base_window12to24_192to384_22kft1k.pt`

這是我們最終拿來預測資料的模型。


## 7. 預測 `orchid_private_set.zip` 和 `orchid_public_set.zip` 中的資料，執行 `inference.py`

最後一個步驟就是要用訓練好的模型來對 public_set 和 private_set 進行預測

首先，先下載 [orchid_private_set.zip](https://drive.google.com/file/d/1Qt5jcyZYnoykcwbkCjRpTHCJWf-JB1Vm/view?usp=sharing)、[orchid_public_set.zip](https://drive.google.com/file/d/18VYedKncZwsru5NgVFDTtHpRZAHf2-zE/view?usp=sharing) 和 [submission_template.csv](https://drive.google.com/file/d/1ZYeBeTvHM3OW9hvZV0u7zRNKHyUH9LWf/view?usp=sharing) (如果已經載過可以略過)



將 `orchid_private_set.zip` 和 `orchid_public_set.zip` 放到 `orchid_competition` 資料夾中

```bash
   orchid_competition
   ├── orchid_private_set.zip
   ├── orchid_public_set.zip
   ├── submission_template.csv
   │ ...

```

接著創一個新的資料夾，取名為 `test_dataset` (也可以取別的名字，但一定要將 `orchid_private_set.zip` 和 `orchid_public_set.zip` 解壓縮到同個資料夾中)
將 `orchid_private_set.zip` 和 `orchid_public_set.zip` 解壓縮到 `test_dataset`

兩個檔案的解壓縮為 
`orchid_private_set.zip`: `Y8vBt&e*AAZ5GREL3#gA9i9j3A`
`orchid_public_set.zip`: `sxRHRQmzmRw8TS!X4Kz23oRvg@`

最後直接執行 inference.py 即可 (如果有改資料夾名稱記得 `test_dataset` 要換成自己設定的)

    $inference.py --IMAGE_DIR <test_dataset>

成功跑完檔案後會產生 `swinv2_submission.csv`

以上就是整個執行的流程。


