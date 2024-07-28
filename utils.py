# AIGC_text_detector/utils.py

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

# 请手动更改nltk_data路径
nltk.data.path.append(r'C:\Users\husuuux\.conda\envs\ella1\nltk_data')


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    return ' '.join(tokens)


def extract_features(text):
    words = text.split()
    num_words = len(words)
    num_sentences = len(nltk.sent_tokenize(text))
    avg_word_length = sum(len(word) for word in words) / num_words if num_words > 0 else 0
    unique_words = set(words)
    vocabulary_richness = len(unique_words) / num_words if num_words > 0 else 0
    pos_tags = nltk.pos_tag(words)
    noun_count = sum(1 for tag in pos_tags if tag[1].startswith('N'))
    verb_count = sum(1 for tag in pos_tags if tag[1].startswith('V'))
    adjective_count = sum(1 for tag in pos_tags if tag[1].startswith('J'))
    return {
        'num_words': num_words,
        'num_sentences': num_sentences,
        'avg_word_length': avg_word_length,
        'vocabulary_richness': vocabulary_richness,
        'noun_count': noun_count,
        'verb_count': verb_count,
        'adjective_count': adjective_count
    }


class LogRegWithMetrics(LogisticRegression):
    def __init__(self, max_iter=200, **kwargs):
        super().__init__(max_iter=max_iter, **kwargs)
        self.loss_history = []
        self.acc_history = []
        self.intermediate_models = []

    def fit(self, X_train, y_train):
        for i in range(1, self.max_iter + 1):
            temp_model = LogisticRegression(max_iter=i)
            temp_model.fit(X_train, y_train)
            self.intermediate_models.append(temp_model)
            y_pred_prob = temp_model.predict_proba(X_train)
            y_pred = temp_model.predict(X_train)
            self.loss_history.append(log_loss(y_train, y_pred_prob))
            self.acc_history.append(accuracy_score(y_train, y_pred))
        # Keep the final model parameters
        super().fit(X_train, y_train)
        return self

