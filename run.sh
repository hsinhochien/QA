#!/bin/bash

# 檢查是否有足夠的參數輸入
if [ "$#" -ne 3 ]; then
    echo "Usage: bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv"
    exit 1
fi

# 設定變數
CONTEXT_PATH="${1}"
TEST_PATH="${2}"
OUTPUT_PATH="${3}"

# 執行 inference.py 並傳遞參數
echo "Running inference with context: $CONTEXT_PATH, test: $TEST_PATH, output: $OUTPUT_PATH"
python3 inference.py --context "$CONTEXT_PATH" --test "$TEST_PATH" --output "$OUTPUT_PATH"
