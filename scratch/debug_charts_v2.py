
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import json

BASE_DIR = "/Users/mohitjain/Desktop/flask-healthcare"
DATA_DIR = os.path.join(BASE_DIR, 'data')

symptom_df   = pd.read_csv(os.path.join(DATA_DIR, 'DiseaseAndSymptoms.csv'))

def chart_json_v1(fig):
    return fig.to_plotly_json()

def chart_json_v2(fig):
    return json.loads(pio.to_json(fig))

# 1. Disease distribution
dc = symptom_df['Disease'].value_counts().reset_index()
dc.columns = ['disease', 'count']
fig1 = px.bar(dc.head(15), x='disease', y='count')

print("Testing v1 (to_plotly_json):")
try:
    c1 = chart_json_v1(fig1)
    json.dumps(c1)
    print("v1: JSON Safe")
except Exception as e:
    print(f"v1: JSON ERROR: {e}")

print("\nTesting v2 (pio.to_json):")
try:
    c2 = chart_json_v2(fig1)
    json.dumps(c2)
    print("v2: JSON Safe")
except Exception as e:
    print(f"v2: JSON ERROR: {e}")
