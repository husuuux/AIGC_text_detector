# AIGC_text_detector/test.py

import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from utils import preprocess_text, extract_features

project_root = os.path.dirname(os.path.abspath(__file__))
folder1 = os.path.join(project_root, 'data', 'MIT Physics pilot')
folder2 = os.path.join(project_root, 'data', 'Chat4.0 MIT Physics pilot')

data = []
labels = []

for filename in os.listdir(folder1):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder1, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)
            labels.append(0)

for filename in os.listdir(folder2):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder2, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)
            labels.append(1)

preprocessed_data = [preprocess_text(text) for text in data]

features = [extract_features(text) for text in preprocessed_data]

df = pd.DataFrame(features)
df['text'] = preprocessed_data
df['label'] = labels

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(df['text'])

X = pd.concat([df.drop(['text', 'label'], axis=1).reset_index(drop=True), pd.DataFrame(X_tfidf.toarray())], axis=1)
y = df['label']

X.columns = X.columns.astype(str)

# Load the model and predict on new data
model = LogisticRegression(max_iter=500)
model.fit(X, y)

new_folder1 = os.path.join(project_root, 'data', 'MIT Physics pilot')
new_folder2 = os.path.join(project_root, 'data', 'Chat4.0 MIT Physics pilot')

new_data = []
new_labels = []

for filename in os.listdir(new_folder1):
    if filename.endswith('.txt'):
        file_path = os.path.join(new_folder1, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            new_data.append(text)
            new_labels.append(0)  # 0 represents 人工写作

for filename in os.listdir(new_folder2):
    if filename.endswith('.txt'):
        file_path = os.path.join(new_folder2, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            new_data.append(text)
            new_labels.append(1)  # 1 represents AI生成

new_preprocessed_data = [preprocess_text(text) for text in new_data]

new_features = [extract_features(text) for text in new_preprocessed_data]

new_df = pd.DataFrame(new_features)
new_df['text'] = new_preprocessed_data

new_X_tfidf = vectorizer.transform(new_df['text'])

new_X = pd.concat([new_df.drop(['text'], axis=1).reset_index(drop=True), pd.DataFrame(new_X_tfidf.toarray())], axis=1)

new_X.columns = new_X.columns.astype(str)

predictions = model.predict(new_X)

for i, prediction in enumerate(predictions):
    print(f"File {i+1}: {'AI生成' if prediction == 1 else '人工写作'}")

accuracy = accuracy_score(new_labels, predictions)
print(f'Accuracy: {accuracy:.4f}')

print("Classification Report:")
print(classification_report(new_labels, predictions))
