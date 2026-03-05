import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = pd.read_csv('/mnt/user-data/uploads/patient_data.csv')

# Rename first column (it was 'C' instead of 'Gender')
data.rename(columns={'C': 'Gender'}, inplace=True)

# Clean up
data['Stages'] = data['Stages'].str.strip()
data['NoseBleeding'] = data['NoseBleeding'].str.strip()

# Normalize stage labels
stage_map_clean = {
    'HYPERTENSION (Stage-1)': 'HYPERTENSION (Stage-1)',
    'HYPERTENSION (Stage-2)': 'HYPERTENSION (Stage-2)',
    'HYPERTENSION (Stage-2).': 'HYPERTENSION (Stage-2)',
    'HYPERTENSIVE CRISIS': 'HYPERTENSIVE CRISIS',
    'HYPERTENSIVE CRISI': 'HYPERTENSIVE CRISIS',
    'NORMAL': 'NORMAL'
}
data['Stages'] = data['Stages'].map(stage_map_clean)
data = data.dropna(subset=['Stages'])

# Encode features
encodings = {
    'Gender': {'Male': 0, 'Female': 1},
    'Age': {'18-34': 1, '35-50': 2, '51-64': 3, '65+': 4},
    'History': {'Yes': 1, 'No': 0},
    'Patient': {'Yes': 1, 'No': 0},
    'TakeMedication': {'Yes': 1, 'No': 0},
    'Severity': {'Mild': 0, 'Moderate': 1, 'Severe': 2},
    'BreathShortness': {'Yes': 1, 'No': 0},
    'VisualChanges': {'Yes': 1, 'No': 0},
    'NoseBleeding': {'Yes': 1, 'No': 0},
    'Whendiagnoused': {'<1 Year': 1, '1 - 5 Years': 2, '>5 Years': 3},
    'Systolic': {'100 - 110': 0, '111 - 120': 1, '121 - 130': 2, '130+': 3},
    'Diastolic': {'70 - 80': 0, '81 - 90': 1, '91 - 100': 2, '100+': 3},
    'ControlledDiet': {'Yes': 1, 'No': 0},
}

stage_label = {
    'NORMAL': 0,
    'HYPERTENSION (Stage-1)': 1,
    'HYPERTENSION (Stage-2)': 2,
    'HYPERTENSIVE CRISIS': 3
}

for col, mapping in encodings.items():
    data[col] = data[col].map(mapping)

data['Stages_enc'] = data['Stages'].map(stage_label)
data = data.dropna()

features = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication', 'Severity',
            'BreathShortness', 'VisualChanges', 'NoseBleeding', 'Whendiagnoused',
            'Systolic', 'Diastolic', 'ControlledDiet']

X = data[features]
y = data['Stages_enc']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc*100:.2f}%")

joblib.dump(model, '/home/claude/hypertension-app/logreg_model.pkl')
print("Model saved as logreg_model.pkl")
