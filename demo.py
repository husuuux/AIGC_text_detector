# AIGC_text_detector/demo.py

import os
import joblib
import pandas as pd
from AIGC_text_detector import preprocess_text, extract_features

project_root = os.path.dirname(os.path.abspath(__file__))
models_folder = os.path.join(project_root, 'models', 'trained_model')
model_path = os.path.join(models_folder, 'logistic_regression_model.pkl')
vectorizer_path = os.path.join(models_folder, 'tfidf_vectorizer.pkl')

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

data_folder = os.path.join(project_root, 'data')
demo_files = ['demo1.txt', 'demo2.txt']

data = []
filenames = []

for demo_file in demo_files:
    file_path = os.path.join(data_folder, demo_file)
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)
            filenames.append(demo_file)
    else:
        print(f"File {demo_file} not found in data directory.")

preprocessed_data = [preprocess_text(text) for text in data]

features = [extract_features(text) for text in preprocessed_data]

df = pd.DataFrame(features)
df['text'] = preprocessed_data
df['filename'] = filenames

X_tfidf = vectorizer.transform(df['text'])

X = pd.concat([df.drop(['text', 'filename'], axis=1).reset_index(drop=True), pd.DataFrame(X_tfidf.toarray())], axis=1)

X.columns = X.columns.astype(str)

predictions = model.predict(X)

for filename, prediction in zip(filenames, predictions):
    label = 'AI生成' if prediction == 1 else '人工写作'
    print(f"File \"{filename}\": {label}")
