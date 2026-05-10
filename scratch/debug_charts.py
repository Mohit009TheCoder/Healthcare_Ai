
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json

BASE_DIR = "/Users/mohitjain/Desktop/flask-healthcare"
DATA_DIR = os.path.join(BASE_DIR, 'data')

diabetes_df  = pd.read_csv(os.path.join(DATA_DIR, 'diabetes_prediction.csv'))
heart_df     = pd.read_csv(os.path.join(DATA_DIR, 'heart_disease_prediction.csv'))
symptom_df   = pd.read_csv(os.path.join(DATA_DIR, 'DiseaseAndSymptoms.csv'))
drug_df      = pd.read_csv(os.path.join(DATA_DIR, 'drug_review.csv'))

COLORS = {
    'primary': '#00d4ff', 'success': '#00ff9d', 'danger': '#ff4d6d',
    'warning': '#ffb703', 'purple': '#a78bfa', 'bg': '#060a14',
    'card': '#0d1424', 'text': '#e2e8f0',
}

def chart_json(fig):
    return fig.to_plotly_json()

charts = {}

# 1. Disease distribution
dc = symptom_df['Disease'].value_counts().reset_index()
dc.columns = ['disease', 'count']
fig1 = px.bar(dc.head(15), x='disease', y='count')
charts['disease_dist'] = chart_json(fig1)

# 2. Top symptoms
all_symptoms_list = []
for col in symptom_df.columns[1:]:
    all_symptoms_list.extend(symptom_df[col].dropna().tolist())
symptom_counts = pd.Series(all_symptoms_list).value_counts().head(15)
fig2 = px.bar(x=symptom_counts.index, y=symptom_counts.values)
charts['sym_heatmap'] = chart_json(fig2)

# 3. Glucose distribution
fig3 = go.Figure()
for diag, color in [('Diabetes', COLORS['danger']), ('No Diabetes', COLORS['success'])]:
    sub = diabetes_df[diabetes_df['diagnosis'] == diag]['glucose_level']
    fig3.add_trace(go.Histogram(x=sub, name=diag))
charts['glucose_dist'] = chart_json(fig3)

# 4. Age distribution
fig4 = go.Figure()
for diag, color in [('Diabetes', COLORS['danger']), ('No Diabetes', COLORS['success'])]:
    sub = diabetes_df[diabetes_df['diagnosis'] == diag]['age']
    fig4.add_trace(go.Violin(y=sub, name=diag))
charts['age_violin'] = chart_json(fig4)

# 5. Heart risk
risk_cols = ['smoking', 'diabetes_history', 'obesity']
yes_pct = [round((heart_df[heart_df['diagnosis'] == 'Heart Disease'][col] == 'Yes').mean() * 100, 1) for col in risk_cols]
no_pct  = [round((heart_df[heart_df['diagnosis'] == 'No Heart Disease'][col] == 'Yes').mean() * 100, 1) for col in risk_cols]
fig5 = go.Figure()
fig5.add_trace(go.Bar(name='Heart Disease', x=risk_cols, y=yes_pct))
fig5.add_trace(go.Bar(name='No Heart Disease', x=risk_cols, y=no_pct))
charts['heart_risk'] = chart_json(fig5)

# 6. BMI vs Glucose
fig6 = px.scatter(diabetes_df, x='bmi', y='glucose_level', color='diagnosis')
charts['bmi_glucose'] = chart_json(fig6)

# 7. Drug rating
top_drugs = drug_df.groupby('medicine_name')['rating'].mean().nlargest(10).reset_index()
fig7 = px.bar(top_drugs, x='medicine_name', y='rating')
charts['drug_rating'] = chart_json(fig7)

# 8. Model accuracy
diab_acc, heart_acc, sym_acc = 0.998, 1.0, 1.0
fig8 = go.Figure(go.Bar(x=['D', 'H', 'S'], y=[diab_acc*100, heart_acc*100, sym_acc*100]))
charts['model_acc'] = chart_json(fig8)

def check_json_safe(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError as e:
        return str(e)

for name, chart in charts.items():
    res = check_json_safe(chart)
    if res == True:
        print(f"Chart {name}: JSON Safe")
    else:
        print(f"Chart {name}: JSON ERROR: {res}")

# Check data types in a sample chart (heart_risk y values)
print(f"heart_risk trace 0 y types: {[type(v) for v in charts['heart_risk']['data'][0]['y']]}")
