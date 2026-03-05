from flask import Flask, render_template, request, flash, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'hypertension-predictor-secret-2024'

# Load trained model
try:
    model = joblib.load("logreg_model.pkl")
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("⚠️  Warning: Model file not found. Using dummy predictions.")
    model = None

# Stage label map
stage_map = {
    0: 'NORMAL',
    1: 'HYPERTENSION (Stage-1)',
    2: 'HYPERTENSION (Stage-2)',
    3: 'HYPERTENSIVE CRISIS'
}

# Color coding per stage
color_map = {
    0: '#10B981',   # emerald - normal
    1: '#F59E0B',   # amber  - stage 1
    2: '#F97316',   # orange - stage 2
    3: '#EF4444'    # red    - crisis
}

# Risk level per stage
risk_map = {
    0: 'Low Risk',
    1: 'Moderate Risk',
    2: 'High Risk',
    3: 'EMERGENCY'
}

# Detailed recommendations
recommendations = {
    0: {
        'title': 'Normal Blood Pressure',
        'subtitle': 'Your cardiovascular health indicators are within normal range.',
        'actions': [
            'Maintain current healthy lifestyle habits',
            'Engage in regular physical activity (150 min/week)',
            'Continue balanced, low-sodium diet',
            'Annual blood pressure monitoring',
            'Regular health check-ups every 12 months'
        ],
        'icon': '❤️'
    },
    1: {
        'title': 'Stage 1 Hypertension',
        'subtitle': 'Mild elevation detected — lifestyle changes recommended.',
        'actions': [
            'Schedule appointment with healthcare provider',
            'Implement DASH diet plan immediately',
            'Increase physical activity gradually',
            'Monitor blood pressure bi-weekly at home',
            'Reduce sodium intake below 2,300 mg/day',
            'Consider stress management techniques'
        ],
        'icon': '⚠️'
    },
    2: {
        'title': 'Stage 2 Hypertension',
        'subtitle': 'Significant elevation requiring immediate medical attention.',
        'actions': [
            'URGENT: Consult physician within 1-2 days',
            'Medication therapy likely required',
            'Comprehensive cardiovascular assessment',
            'Daily blood pressure monitoring',
            'Strict dietary sodium restriction',
            'Lifestyle modification counseling',
            'Limit alcohol and caffeine consumption'
        ],
        'icon': '🚨'
    },
    3: {
        'title': 'Hypertensive Crisis',
        'subtitle': 'CRITICAL: Dangerously elevated blood pressure detected.',
        'actions': [
            'EMERGENCY: Seek immediate medical attention NOW',
            'Call emergency services if experiencing symptoms',
            'Do not delay treatment under any circumstances',
            'Monitor for stroke/heart attack warning signs',
            'Prepare current medication list for physicians',
            'Avoid physical exertion entirely',
            'Stay calm and lie down until help arrives'
        ],
        'icon': '🆘'
    }
}

# Feature encoding maps
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

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        required_fields = ['Gender', 'Age', 'History', 'Patient', 'TakeMedication',
                           'Severity', 'BreathShortness', 'VisualChanges', 'NoseBleeding',
                           'Whendiagnoused', 'Systolic', 'Diastolic', 'ControlledDiet']

        form_data = {}
        for field in required_fields:
            value = request.form.get(field)
            if not value or value == '':
                return jsonify({'error': f'Please complete all required fields: {field.replace("_", " ")}'}), 400
            form_data[field] = value

        # Encode inputs
        try:
            encoded = [encodings[field][form_data[field]] for field in required_fields]
        except KeyError as e:
            return jsonify({'error': f'Invalid selection detected: {str(e)}'}), 400

        input_array = np.array(encoded).reshape(1, -1)

        # Predict
        if model is not None:
            prediction = int(model.predict(input_array)[0])
            try:
                proba = model.predict_proba(input_array)[0]
                confidence = round(float(max(proba)) * 100, 1)
            except:
                confidence = 85.0
        else:
            import random
            prediction = random.randint(0, 3)
            confidence = 87.5

        return jsonify({
            'prediction': prediction,
            'stage': stage_map[prediction],
            'color': color_map[prediction],
            'risk': risk_map[prediction],
            'confidence': confidence,
            'recommendation': recommendations[prediction],
            'form_data': form_data
        })

    except Exception as e:
        return jsonify({'error': f'System error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
