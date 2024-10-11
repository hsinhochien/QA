#!/bin/bash

FOLDER_ID_1="https://drive.google.com/drive/folders/1nuaqs9SUFnc1SRHAdYbRuJ0vPwX5joQn?usp=sharing"
FOLDER_ID_2="https://drive.google.com/drive/folders/1kZ2hy1YVgBAMbP_jJN4wlcDCRIXD7qzQ?usp=sharing"

# 定義下載後存放的目錄
MODEL_DIR_1="./paragraph_selection_model_parameters"
MODEL_DIR_2="./span_selection_model_parameters"

# 檢查第一個模型目錄是否存在，不存在則創建
if [ ! -d "$MODEL_DIR_1" ]; then
    echo "Creating model directory: $MODEL_DIR_1"
    mkdir -p "$MODEL_DIR_1"
fi

# 檢查第二個模型目錄是否存在，不存在則創建
if [ ! -d "$MODEL_DIR_2" ]; then
    echo "Creating model directory: $MODEL_DIR_2"
    mkdir -p "$MODEL_DIR_2"
fi

# 下載第一個Google雲端硬碟資料夾的所有內容
echo "Downloading paragraph selection model parameters from Google Drive..."
gdown --folder "$FOLDER_ID_1" -O "$MODEL_DIR_1"

# 下載第二個Google雲端硬碟資料夾的所有內容
echo "Downloading span selection model parameters from Google Drive..."
gdown --folder "$FOLDER_ID_2" -O "$MODEL_DIR_2"

echo "Both model parameters downloaded to $MODEL_DIR_1 and $MODEL_DIR_2"
