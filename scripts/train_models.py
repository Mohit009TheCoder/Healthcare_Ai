#!/usr/bin/env python3
# train_models.py
"""
Train all ML models once and save them for fast loading.
Run this script: python train_models.py

This eliminates the need to retrain models on every app startup,
reducing startup time from 60 seconds to ~2 seconds.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("HEALTHCARE ML MODEL TRAINING")
print("=" * 70)

# Create models directory
os.makedirs('models', exist_ok=True)
print("\n✅ Models directory ready")

# Load datasets
print("\n📂 Loading datasets...")
DATA_DIR = 'data'
diabetes_df = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_prediction.csv'))
heart_df = pd.read_csv(os.path.join(DATA_DIR, 'heart_disease_prediction.csv'))
symptom_df = pd.read_csv(os.path.join(DATA_DIR, 'DiseaseAndSymptoms.csv'))
print(f"   Diabetes: {len(diabetes_df)} records")
print(f"   Heart: {len(heart_df)} records")
print(f"   Symptoms: {len(symptom_df)} records")

# ═══════════════════════════════════════════════════════════════════════════
# 1. TRAIN DIABETES MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("1. TRAINING DIABETES PREDICTION MODEL")
print("=" * 70)

le_diab = LabelEncoder()
diabetes_clean = diabetes_df.copy()

# Encode binary features
for col in ['frequent_urination', 'excessive_thirst', 'fatigue', 'blurred_vision']:
    if col in diabetes_clean.columns:
        diabetes_clean[col] = le_diab.fit_transform(diabetes_clean[col].astype(str))

diabetes_clean['target'] = (diabetes_clean['diagnosis'] == 'Diabetes').astype(int)

DIAB_FEATURES = ['age', 'glucose_level', 'bmi', 'insulin_level', 'blood_pressure',
                  'pregnancies', 'diabetes_pedigree_function',
                  'frequent_urination', 'excessive_thirst', 'fatigue', 'blurred_vision']

X_diab = diabetes_clean[DIAB_FEATURES]
y_diab = diabetes_clean['target']

X_tr_d, X_te_d, y_tr_d, y_te_d = train_test_split(
    X_diab, y_diab, test_size=0.2, random_state=42
)

# Scale features
scaler_diab = StandardScaler()
X_tr_d_s = scaler_diab.fit_transform(X_tr_d)
X_te_d_s = scaler_diab.transform(X_te_d)

# Train Random Forest model
print("   Training Random Forest...")
diabetes_model = RandomForestClassifier(
    n_estimators=200, 
    random_state=42, 
    max_depth=10, 
    n_jobs=-1
)
diabetes_model.fit(X_tr_d_s, y_tr_d)

# Evaluate
diab_acc = accuracy_score(y_te_d, diabetes_model.predict(X_te_d_s))
print(f"   ✅ Diabetes Model Accuracy: {diab_acc:.3f} ({diab_acc*100:.1f}%)")

# Save
joblib.dump(diabetes_model, 'models/diabetes_model_v1.pkl')
joblib.dump(scaler_diab, 'models/diabetes_scaler_v1.pkl')
joblib.dump(DIAB_FEATURES, 'models/diabetes_features_v1.pkl')
print("   💾 Saved: diabetes_model_v1.pkl, diabetes_scaler_v1.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# 2. TRAIN HEART DISEASE MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("2. TRAINING HEART DISEASE PREDICTION MODEL")
print("=" * 70)

heart_clean = heart_df.copy()

# Encode binary features
bool_cols_h = ['smoking', 'diabetes_history', 'obesity', 'chest_pain',
               'shortness_of_breath', 'dizziness', 'fatigue']
for col in bool_cols_h:
    if col in heart_clean.columns:
        heart_clean[col] = (heart_clean[col] == 'Yes').astype(int)

heart_clean['target'] = (heart_clean['diagnosis'] == 'Heart Disease').astype(int)

HEART_FEATURES = ['age', 'cholesterol', 'resting_blood_pressure', 'maximum_heart_rate',
                   'smoking', 'diabetes_history', 'obesity',
                   'chest_pain', 'shortness_of_breath', 'dizziness', 'fatigue']

X_heart = heart_clean[HEART_FEATURES]
y_heart = heart_clean['target']

X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(
    X_heart, y_heart, test_size=0.2, random_state=42
)

# Scale features
scaler_heart = StandardScaler()
X_tr_h_s = scaler_heart.fit_transform(X_tr_h)
X_te_h_s = scaler_heart.transform(X_te_h)

# Train Gradient Boosting model
print("   Training Gradient Boosting...")
heart_model = GradientBoostingClassifier(n_estimators=200, random_state=42)
heart_model.fit(X_tr_h_s, y_tr_h)

# Evaluate
heart_acc = accuracy_score(y_te_h, heart_model.predict(X_te_h_s))
print(f"   ✅ Heart Disease Model Accuracy: {heart_acc:.3f} ({heart_acc*100:.1f}%)")

# Save
joblib.dump(heart_model, 'models/heart_model_v1.pkl')
joblib.dump(scaler_heart, 'models/heart_scaler_v1.pkl')
joblib.dump(HEART_FEATURES, 'models/heart_features_v1.pkl')
print("   💾 Saved: heart_model_v1.pkl, heart_scaler_v1.pkl")

# ═══════════════════════════════════════════════════════════════════════════
# 3. TRAIN SYMPTOM-DISEASE MODEL
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("3. TRAINING SYMPTOM-DISEASE PREDICTION MODEL")
print("=" * 70)

# Collect all unique symptoms
symptom_cols = [c for c in symptom_df.columns if c != 'Disease']

all_symptoms_raw = set()
for col in symptom_cols:
    vals = symptom_df[col].dropna().unique()
    for v in vals:
        cleaned = str(v).strip().lower().replace(' ', '_')
        if cleaned and cleaned != 'nan':
            all_symptoms_raw.add(cleaned)

ALL_SYMPTOMS = sorted(all_symptoms_raw)
print(f"   Total unique symptoms: {len(ALL_SYMPTOMS)}")

# Build binary feature matrix
def symptoms_to_vector(symptom_list):
    s_set = set(s.strip().lower().replace(' ', '_') for s in symptom_list)
    return [1 if s in s_set else 0 for s in ALL_SYMPTOMS]

rows = []
labels = []
for _, row in symptom_df.iterrows():
    syms = []
    for col in symptom_cols:
        val = str(row[col]).strip()
        if val and val.lower() != 'nan':
            syms.append(val)
    if syms:
        rows.append(symptoms_to_vector(syms))
        labels.append(str(row['Disease']).strip())

X_sym = np.array(rows)
y_sym_raw = np.array(labels)

# Encode labels
sym_le = LabelEncoder()
y_sym = sym_le.fit_transform(y_sym_raw)
DISEASE_NAMES = list(sym_le.classes_)

print(f"   Training matrix: {X_sym.shape[0]} samples × {X_sym.shape[1]} features")
print(f"   Disease classes: {len(DISEASE_NAMES)}")

# Train/test split
X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
    X_sym, y_sym, test_size=0.2, random_state=42, stratify=y_sym
)

# Train ensemble model
print("   Training Ensemble (RF + NB + LR)...")
rf_sym = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
nb_sym = GaussianNB()
lr_sym = LogisticRegression(max_iter=1000, random_state=42, C=1.0)

symptom_model = VotingClassifier(
    estimators=[('rf', rf_sym), ('nb', nb_sym), ('lr', lr_sym)],
    voting='soft'
)
symptom_model.fit(X_tr_s, y_tr_s)

# Evaluate
sym_acc = accuracy_score(y_te_s, symptom_model.predict(X_te_s))
print(f"   ✅ Symptom-Disease Model Accuracy: {sym_acc:.3f} ({sym_acc*100:.1f}%)")

# Build disease-symptom map for confidence scoring
DISEASE_SYM_MAP = {}
for _, _row in symptom_df.iterrows():
    _d = str(_row['Disease']).strip()
    if _d not in DISEASE_SYM_MAP:
        DISEASE_SYM_MAP[_d] = set()
    for _col in symptom_cols:
        _v = str(_row[_col]).strip().lower().replace(' ', '_')
        if _v and _v != 'nan':
            DISEASE_SYM_MAP[_d].add(_v)

# Save
joblib.dump(symptom_model, 'models/symptom_model_v1.pkl')
joblib.dump(sym_le, 'models/symptom_label_encoder_v1.pkl')
joblib.dump(ALL_SYMPTOMS, 'models/all_symptoms_v1.pkl')
joblib.dump(DISEASE_NAMES, 'models/disease_names_v1.pkl')
joblib.dump(DISEASE_SYM_MAP, 'models/disease_symptom_map_v1.pkl')
print("   💾 Saved: symptom_model_v1.pkl, symptom_label_encoder_v1.pkl, etc.")

# ═══════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("✅ ALL MODELS TRAINED AND SAVED SUCCESSFULLY!")
print("=" * 70)
print(f"\n📊 Model Performance Summary:")
print(f"   • Diabetes Model:       {diab_acc*100:.1f}% accuracy")
print(f"   • Heart Disease Model:  {heart_acc*100:.1f}% accuracy")
print(f"   • Symptom-Disease Model: {sym_acc*100:.1f}% accuracy")
print(f"\n💾 Models saved in: ./models/")
print(f"   Total files: {len(os.listdir('models'))}")
print(f"\n🚀 You can now run the Flask app with fast model loading!")
print("=" * 70)
