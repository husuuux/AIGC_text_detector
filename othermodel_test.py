# This file is used to test the other model (English_v2) and compare its results with our model.

import os
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, classification_report

project_root = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(project_root, 'models', 'English_v2')

config = AutoConfig.from_pretrained(model_folder)
model = AutoModelForSequenceClassification.from_pretrained(model_folder, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_folder)

# Length of the longest sequence to be used for the input to the model.
max_length = 512  

folder1 = os.path.join(project_root, 'data', 'MIT Physics pilot')
folder2 = os.path.join(project_root, 'data', 'Chat4.0 MIT Physics pilot')

data = []
labels = []

for filename in os.listdir(folder1):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder1, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            data.append(text)
            labels.append(0)  # 0 represents 人工写作

for filename in os.listdir(folder2):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder2, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read().strip()
            data.append(text)
            labels.append(1)  # 1 represents AI生成

predictions = []

for text in data:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)

    with torch.no_grad():  # Turn off gradients to speed up the prediction
        outputs = model(**inputs)

    prediction = outputs.logits.argmax(dim=-1).item()
    predictions.append(prediction)

accuracy = accuracy_score(labels, predictions)
print(f'Accuracy: {accuracy:.4f}')

print("Classification Report:")
print(classification_report(labels, predictions))
