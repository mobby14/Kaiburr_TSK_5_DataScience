"""
Consumer Complaint Classification

This script downloads the CFPB Consumer Complaint Database, filters for four target
product categories, performs EDA, feature engineering, text pre-processing,
trains multiple classification models (TF-IDF + Logistic Regression, RandomForest,
and an optional HuggingFace transformer), compares their performance, evaluates
and saves the best model, and shows how to predict on new texts.

Notes:
- The full CSV is large (~3GB unzipped). The script by default works on a sampled
  subset for development. For full training increase `SAMPLE_FRAC` or remove sampling.
- You need Python 3.8+, and packages listed below.

References / data source:
- Consumer Complaint Database landing page (download links). See: https://www.consumerfinance.gov/data-research/consumer-complaints/ 
- Direct CSV download (zip): https://files.consumerfinance.gov/ccdb/complaints.csv.zip

"""
import os
import sys
import zipfile
import requests
import io
import pandas as pd
import numpy as np
from collections import Counter

# ML / NLP
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib

# Optional for transformer (HuggingFace)
USE_TRANSFORMER = False

if USE_TRANSFORMER:
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
        import torch
    except Exception as e:
        print("Transformer libraries not installed. Set USE_TRANSFORMER=False or install 'transformers' and 'datasets'.")
        USE_TRANSFORMER = False

# ----------------------------
# Config
# ----------------------------
ZIP_URL = 'https://files.consumerfinance.gov/ccdb/complaints.csv.zip'
LOCAL_ZIP = 'complaints.csv.zip'
CSV_NAME = 'complaints.csv'  # inside zip
SAMPLE_FRAC = 0.05  # change to None or 1.0 to use full dataset (requires lots of RAM/time)
RANDOM_STATE = 42
TARGET_MAP = {
    'Credit reporting, credit repair services, or other': 0,
    'Debt collection': 1,
    'Consumer Loan': 2,
    'Mortgage': 3
}
# Some datasets may use slightly different product names. We'll map by startswith/includes too.
ALT_MATCH = {
    'credit report': 0,
    'debt collection': 1,
    'consumer loan': 2,
    'mortgage': 3
}

SAVE_DIR = 'models'
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Helper: download and load (streamed)
# ----------------------------

def download_and_extract_csv(url=ZIP_URL, local_zip=LOCAL_ZIP, csv_name=CSV_NAME):
    if not os.path.exists(local_zip):
        print('Downloading', url)
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_zip, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    else:
        print('Using cached', local_zip)

    # open zip and load csv in chunks to avoid memory spikes
    with zipfile.ZipFile(local_zip, 'r') as z:
        if csv_name not in z.namelist():
            # try to find the first csv inside
            csv_candidates = [n for n in z.namelist() if n.lower().endswith('.csv')]
            if not csv_candidates:
                raise FileNotFoundError('No CSV found inside zip')
            csv_name = csv_candidates[0]
        print('Reading', csv_name)
        with z.open(csv_name) as f:
            # read in chunks and collect into dataframe (careful with memory)
            df = pd.read_csv(f, dtype=str, low_memory=False)
    return df

# ----------------------------
# Main pipeline
# ----------------------------

def main():
    print('Step 1: Download & load (this may take a while and lots of RAM)')
    df = download_and_extract_csv()
    print('Loaded rows:', len(df))

    # Keep relevant columns
    cols_needed = ['product', 'sub_product', 'issue', 'sub_issue', 'consumer_complaint_narrative', 'company_public_response', 'company', 'state', 'zip_code', 'date_received']
    available = [c for c in cols_needed if c in df.columns]
    df = df[available].copy()

    # Basic EDA
    print('\nStep 2: EDA (basic)')
    print('Columns available:', df.columns.tolist())
    print('Missing narrative count:', df['consumer_complaint_narrative'].isna().sum() if 'consumer_complaint_narrative' in df.columns else 'N/A')
    print('Top products:')
    print(df['product'].value_counts().head(20))

    # Map products to our 4 classes
    def map_product(prod):
        if pd.isna(prod):
            return None
        prod_l = prod.lower()
        for key, val in ALT_MATCH.items():
            if key in prod_l:
                return val
        # exact mapping
        if prod in TARGET_MAP:
            return TARGET_MAP[prod]
        return None

    df['label'] = df['product'].map(map_product)
    df = df[df['label'].notna()].copy()
    df['label'] = df['label'].astype(int)
    print('\nFiltered dataset size (only 4 categories):', len(df))
    print('Label distribution:')
    print(df['label'].value_counts(normalize=True))

    # Use narrative as text; if narrative missing, try company_response or issue
    def build_text(row):
        parts = []
        if 'consumer_complaint_narrative' in row and pd.notna(row['consumer_complaint_narrative']):
            parts.append(str(row['consumer_complaint_narrative']))
        if 'company_public_response' in row and pd.notna(row['company_public_response']):
            parts.append(str(row['company_public_response']))
        parts.append(str(row.get('issue', '')))
        parts.append(str(row.get('sub_issue', '')))
        return ' '.join([p for p in parts if p and p != 'nan'])

    df['text'] = df.apply(build_text, axis=1)
    df = df[df['text'].str.len() > 10]

    # Optional sampling (to keep runtime and memory reasonable)
    if SAMPLE_FRAC and SAMPLE_FRAC < 1.0:
        df = df.sample(frac=SAMPLE_FRAC, random_state=RANDOM_STATE)
        print('Sampled dataset size:', len(df))

    # Train/test split
    X = df['text'].values
    y = df['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # ----------------------------
    # Text pre-processing + model pipelines
    # ----------------------------
    print('\nStep 3: Text pre-processing and model selection')
    # We'll use TF-IDF (word + ngram) as common feature extractor
    vect = TfidfVectorizer(max_features=50000, ngram_range=(1,2), min_df=5)

    # Model 1: Logistic Regression (strong baseline)
    pipe_lr = Pipeline([
        ('tfidf', vect),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', solver='saga'))
    ])

    # Model 2: Random Forest (bagging)
    pipe_rf = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=10)),
        ('clf', RandomForestClassifier(n_estimators=200, n_jobs=-1, class_weight='balanced', random_state=RANDOM_STATE))
    ])

    # Train models
    print('Training Logistic Regression...')
    pipe_lr.fit(X_train, y_train)
    print('Training Random Forest...')
    pipe_rf.fit(X_train, y_train)

    # Evaluate
    print('\nStep 4: Compare model performance on test set')
    models = {'LogisticRegression': pipe_lr, 'RandomForest': pipe_rf}
    results = {}
    for name, m in models.items():
        preds = m.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1_macro = f1_score(y_test, preds, average='macro')
        print('\nModel:', name)
        print('Accuracy:', acc)
        print('Macro F1:', f1_macro)
        print(classification_report(y_test, preds, digits=4))
        results[name] = {'accuracy': acc, 'f1_macro': f1_macro}

    # Optionally train transformer
    if USE_TRANSFORMER:
        print('\nStep 5: Fine-tuning transformer (this is optional and requires GPUs to train quickly)')
        # Map labels to 0..n-1 already done
        num_labels = len(np.unique(y_train))
        model_name = 'distilbert-base-uncased'
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Prepare dataset for HF Trainer (simple wrapper)
        import datasets
        def tokenize_fn(ex):
            return tokenizer(ex['text'], truncation=True, padding='max_length', max_length=256)

        train_ds = datasets.Dataset.from_dict({'text': list(X_train), 'label': list(y_train)})
        test_ds = datasets.Dataset.from_dict({'text': list(X_test), 'label': list(y_test)})
        train_ds = train_ds.map(tokenize_fn, batched=True)
        test_ds = test_ds.map(tokenize_fn, batched=True)
        train_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])
        test_ds.set_format(type='torch', columns=['input_ids','attention_mask','label'])

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        training_args = TrainingArguments(output_dir='./tf_out', per_device_train_batch_size=8, per_device_eval_batch_size=8, num_train_epochs=2, evaluation_strategy='epoch', save_strategy='no')
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=test_ds)
        trainer.train()
        # Evaluate
        preds_logits = trainer.predict(test_ds).predictions
        preds = np.argmax(preds_logits, axis=1)
        print('Transformer results:')
        print(classification_report(y_test, preds))

    # Save best model (choose by f1_macro)
    best_name = max(results.items(), key=lambda x: x[1]['f1_macro'])[0]
    print('\nBest model on metrics:', best_name)
    best_model = models[best_name]
    joblib.dump(best_model, os.path.join(SAVE_DIR, f'{best_name}.joblib'))
    print('Saved model to', os.path.join(SAVE_DIR, f'{best_name}.joblib'))

    # ----------------------------
    # Prediction function
    # ----------------------------
    def predict_texts(texts, model=best_model):
        preds = model.predict(texts)
        # map back to label names
        inv_map = {v:k for k,v in TARGET_MAP.items()}
        # fallback names
        label_names = {0: 'Credit reporting/repair/other', 1: 'Debt collection', 2: 'Consumer Loan', 3: 'Mortgage'}
        return [label_names.get(int(p), str(p)) for p in preds]

    # Demo predictions
    demo_texts = [
        'I found a wrong account on my credit report and the credit bureau will not correct it.',
        'I keep getting calls from a debt collector about a loan I paid.',
        'My auto loan monthly payment was miscalculated and I was charged wrong interest.',
        'I applied for a mortgage and the company denied my application without explanation.'
    ]
    print('\nStep 6: Predictions (demo)')
    print(predict_texts(demo_texts))


if __name__ == '__main__':
    main()
