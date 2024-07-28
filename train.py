# AIGC_text_detector/train.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from AIGC_text_detector import preprocess_text, extract_features, LogRegWithMetrics
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib


project_root = os.path.dirname(os.path.abspath(__file__))
folder1 = os.path.join(project_root, 'data', 'MIT_Dept. of Economicspilot')
folder2 = os.path.join(project_root, 'data', 'chat4.0 Economicspilot')

data = []
labels = []

for filename in os.listdir(folder1):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder1, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)
            labels.append(1)  # 1 represents AI生成

for filename in os.listdir(folder2):
    if filename.endswith('.txt'):
        file_path = os.path.join(folder2, filename)
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)
            labels.append(0)  # 0 represents 人工写作

preprocessed_data = [preprocess_text(text) for text in data]

features = [extract_features(text) for text in preprocessed_data]

df = pd.DataFrame(features)
df['text'] = preprocessed_data
df['label'] = labels

# Use TF-IDF to convert text into numerical features
vectorizer = TfidfVectorizer(max_features=1000)  # limit to 1000 features for memory reasons
X_tfidf = vectorizer.fit_transform(df['text'])

X = pd.concat([df.drop(['text', 'label'], axis=1).reset_index(drop=True), pd.DataFrame(X_tfidf.toarray())], axis=1)
y = df['label']

X.columns = X.columns.astype(str)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LogRegWithMetrics(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

loss_history = model.loss_history
acc_history = model.acc_history

# picture of loss and accuracy curve
epochs = range(1, len(loss_history) + 1)

plt.figure(figsize=(18, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, loss_history, label='Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 3, 2)
plt.plot(epochs, acc_history, label='Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')

feature_names = list(X.columns)
feature_importance = model.coef_[0]
# visualize feature importance -- most important features on top
indices = np.argsort(np.abs(feature_importance))[::-1]
top_features = 20  # number of top features to show

plt.subplot(1, 3, 3)
plt.title("Top 20 Feature Importances", fontsize=16)
plt.barh(range(top_features), feature_importance[indices][:top_features], align='center')
plt.yticks(range(top_features), [feature_names[i] for i in indices[:top_features]])
plt.xlabel("Feature Importance", fontsize=14)
plt.gca().invert_yaxis()

plt.tight_layout()
plt.show()

# Save the TF-IDF Vectorizer and Logistic Regression Model
cache_folder = os.path.join(project_root, 'models', 'runs')
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)

vectorizer_path = os.path.join(cache_folder, 'tfidf_vectorizer.pkl')
model_path = os.path.join(cache_folder, 'logistic_regression_model.pkl')

joblib.dump(vectorizer, vectorizer_path)
joblib.dump(model, model_path)
print("Vectorizer and Model saved.")