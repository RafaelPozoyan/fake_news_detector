import os
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split

from utils import preprocess_text, combine_features, LABEL_MAPPING


def load_and_prepare(data_dir: str) -> pd.DataFrame:
    train_bodies = pd.read_csv(os.path.join(data_dir, 'train_bodies.csv'))
    train_stances = pd.read_csv(os.path.join(data_dir, 'train_stances.csv'))

    data = train_stances.merge(train_bodies, on='Body ID', how='left')
    data = data.dropna(subset=['Headline', 'articleBody', 'Stance']).reset_index(drop=True)

    data['headline_clean'] = data['Headline'].apply(preprocess_text)
    data['body_clean'] = data['articleBody'].apply(preprocess_text)
    data['combined_text'] = data.apply(
        lambda r: combine_features(r['headline_clean'], r['body_clean'], max_body_words=100), axis=1
    )
    data = data[data['combined_text'].str.len() > 10].reset_index(drop=True)

    # Map stance to binary label: agree -> 1 (real), disagree -> 0 (fake)
    label_map = {'agree': 1, 'disagree': 0}
    data['label'] = data['Stance'].map(label_map)
    data = data.dropna(subset=['label']).reset_index(drop=True)
    data['label'] = data['label'].astype(int)
    return data


def main():
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)

    print('[1/4] Load and preprocess data')
    df = load_and_prepare(data_dir)
    X = df['combined_text']
    y = df['label']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print('[2/4] Vectorize')
    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.85,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_val_vec = vectorizer.transform(X_val)

    print('[3/4] Train model')
    model = LogisticRegression(max_iter=1000, C=1.0, class_weight='balanced', random_state=42)
    model.fit(X_train_vec, y_train)

    print('[4/4] Evaluate')
    y_pred = model.predict(X_val_vec)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted')
    print(f'Accuracy: {acc:.4f}\nF1-weighted: {f1:.4f}')
    print(classification_report(y_val, y_pred, target_names=['Фейк', 'Реальная'], digits=4))

    with open(os.path.join(models_dir, 'fake_news_detector.pkl'), 'wb') as f:
        pickle.dump(model, f)
    with open(os.path.join(models_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(os.path.join(models_dir, 'label_mapping.pkl'), 'wb') as f:
        pickle.dump(LABEL_MAPPING, f)

    print('Saved artifacts to models/')


if __name__ == '__main__':
    main()


