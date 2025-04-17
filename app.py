from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from flask import render_template

# Initialize Flask app
app = Flask(__name__)

# Load and prepare data
df = pd.read_csv('./Disease_symptom_and_patient_profile_dataset.csv')

input_features = ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing',
                  'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']

# Encode categorical columns
label_encoders = {}
for col in input_features + ['Outcome Variable']:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

X = df[input_features]
y = df['Outcome Variable']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON input
        data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Encode using trained label_encoders
        for col in input_df.columns:
            if col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col].astype(str))

        # Predict
        prediction = model.predict(input_df)
        outcome = label_encoders['Outcome Variable'].inverse_transform(prediction)

        return jsonify({'predicted_outcome': outcome[0]})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
      app.run(debug=True, port=5005)
