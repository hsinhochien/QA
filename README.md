# Chinese Extractive Question Answering
## mc.ipynb
This file fine-tunes the hfl/chinese-roberta-wwm-ext model for a multiple-choice task. 
### Prerequisite
```pip install transformers datasets pandas evaluate```
### Steps to train the model
1. Prepare data <br>
Ensure you have the files context.json, train.json, valid.json ready.
2. Preprocess the data <br>
The ```prepare_data_for_mc``` function converts each question and its corresponding paragraph options into the multiple-choice format needed for the model.
3. Tokenize the data <br>
The ```preprocess_function``` uses the tokenizer from hfl/chinese-roberta-wwm-ext to tokenize both the question and paragraph options.
4. Train the Model <br>
Run all cells and the best model parameters will be saved to ./finetuned_mc.

## qa-lert-l.ipynb
This file fine-tunes the hfl/chinese-lert-large model for a question-answering task. 
### Prerequisite
```pip install transformers datasets pandas scikit-learn```
### Steps to train the model
1. Prepare data <br>
Ensure you have the files context.json, train.json, valid.json ready.
2. Preprocess the data <br>
The ```preprocess_function``` is responsible for tokenizing the questions and paragraphs, while also calculating the start and end positions of the answer within the context.
3. Train the Model <br>
Run all cells and the best model parameters will be saved to ./finetuned_qa.

## inference.py
This file is used to perform inference on test.json using the two models I trained.

## download.sh
Run this file by executing ```bash ./download.sh```. <br>
This will download the folders ./finetuned_mc and ./finetuned_qa, which contain the model parameters for my models.

## run.sh
Run this file by executing ```bash ./run.sh /path/to/context.json /path/to/test.json /path/to/pred/prediction.csv```. <br>
This will execute the inference.py file.
