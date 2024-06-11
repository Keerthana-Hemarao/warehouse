from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the model and encoders
model = joblib.load('logistic_regression_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
target_encoder = joblib.load('target_encoder.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    
    # Convert 'Creation Date' to datetime
    input_df['Creation Date'] = pd.to_datetime(input_df['Creation Date'])
    input_df = input_df.drop(['Creation Date'], axis=1)
    
    # Encode categorical input features
    for column in input_df.select_dtypes(include=['object']).columns:
        input_df[column] = label_encoders[column].transform(input_df[column])
    
    # Predict the final result warehouse
    predicted_numeric = model.predict(input_df)
    predicted_label = target_encoder.inverse_transform(predicted_numeric)
    
    return jsonify({'Predicted Final Result warehouse': predicted_label[0]})

if __name__ == '__main__':
    app.run(debug=True)
