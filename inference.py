from transformers import AutoTokenizer, AutoModelForMultipleChoice, pipeline
import torch
import csv
import json
from tqdm import tqdm
import argparse

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def preprocess_data(data, paragraphs):
    samples = []
    for item in data:
        question = item['question']
        for para_id in item['paragraphs']:
            para_text = paragraphs[para_id]
            
            input_text = f"{question} [SEP] {para_text}"
            
            samples.append({
                'id': item['id'],  
                'question': question,
                'paragraph_id': para_id,
                'input_text': input_text
            })
    return samples

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference for QA model")
    parser.add_argument('--context', required=True, help='Path to context.json')
    parser.add_argument('--test', required=True, help='Path to test.json')
    parser.add_argument('--output', required=True, help='Path to output prediction.csv')
    return parser.parse_args()

# 解析命令行參數
args = parse_args()

# 使用傳入的路徑讀取資料
paragraphs = load_data(args.context)
test_data = load_data(args.test)

## ---------- multiple-choice
def transform_data(data):
    mc_data = []

    for entry in data:
        question = entry['question']
        
        # 將每個段落（作為選項）轉換為字符串，並確保有4個選項
        options = [str(paragraphs[para_id]) for para_id in entry['paragraphs']]
            
        mc_data.append({
            'id': entry['id'],
            'question': question,
            'paragraphs': entry['paragraphs'],
            'candidate0': options[0],  # 第一个段落
            'candidate1': options[1],  # 第二个段落
            'candidate2': options[2],  # 第三个段落
            'candidate3': options[3]   # 第四个段落
        })
    return mc_data

test_data = transform_data(test_data)

tokenizer = AutoTokenizer.from_pretrained("./paragraph_selection_model_parameters")
model = AutoModelForMultipleChoice.from_pretrained("./paragraph_selection_model_parameters")

multiple_choice_results = []

print("Starting multiple choice inference.")
for example in tqdm(test_data, total=len(test_data)):
    inputs = tokenizer([[example['question'], example['candidate0']], [example['question'], example['candidate1']], [example['question'], example['candidate2']], [example['question'], example['candidate3']]], 
                       truncation=True, return_tensors="pt", padding='max_length', max_length=512)
    labels = torch.tensor(0).unsqueeze(0)
    outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
    logits = outputs.logits
    predicted_class = logits.argmax().item()

    multiple_choice_results.append(
        {
            'id': example['id'],
            'predicted_paragraph_id': example['paragraphs'][predicted_class]
        }
    )

## ---------- question-answering
model = "./span_selection_model_parameters"
question_answerer = pipeline("question-answering", model=model, tokenizer=model)

csv_filename = args.output
with open(csv_filename, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    # 寫入標題
    writer.writerow(['id', 'answer'])

    # 遍歷每筆 test_sample 並進行推理
    print("Starting question answering inference.")
    for idx, sample in tqdm(enumerate(multiple_choice_results), total=len(multiple_choice_results)):
        predicted_paragraph_id = sample['predicted_paragraph_id']
        current_id = sample['id']

        # 檢查 predicted_paragraph_id 是否為 None
        if predicted_paragraph_id is None:
            result = None  # 如果沒有預測段落，result 設為 None
        else:
            question = test_data[idx]['question']
            context = paragraphs[predicted_paragraph_id]  # 根據 predicted_paragraph_id 取得段落
            # 使用 question_answerer pipeline 做問答
            result = question_answerer(question=question, context=context)['answer']

        # 寫入資料到 CSV
        writer.writerow([current_id, result])

print(f"Results saved to {csv_filename}")