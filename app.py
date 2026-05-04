import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# Get the absolute path to the project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATE_DIR)
CORS(app)

# --- CONFIGURATION ---
MODEL_DIR = 'models'
DATASET_PATH = 'dataset/CurrentDataset_cleaned_fixed_new.csv'

# --- LOAD MODELS & INFO ---
model = None
scaler = None
model_info = None
sample_df = None

def load_system():
    global model, scaler, model_info, sample_df
    try:
        model = joblib.load(os.path.join(MODEL_DIR, 'best_model.pkl'))
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        
        info_path = os.path.join(MODEL_DIR, 'best_model_info.json')
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                model_info = json.load(f)
        
        if os.path.exists(DATASET_PATH):
            sample_df = pd.read_csv(DATASET_PATH)
            
        print("System components loaded successfully")
    except Exception as e:
        print(f"Error loading system: {e}")

load_system()

# --- ROUTES ---

@app.route('/')
def index():
    cities = []
    if sample_df is not None:
        cities = sorted(sample_df['city'].unique().tolist())
    return render_template('index.html', cities=cities, model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # 1. Prepare raw input dict matching terminal script
        user_data = {
            'city': data.get('city'),
            'ZipCode': int(data.get('ZipCode', 0)),
            'Latitude': float(data.get('Latitude', 0.0)),
            'Longtitude': float(data.get('Longtitude', 0.0)),
            'HospitalDistance': float(data.get('HospitalDistance', 0.0)),
            'Bank/ATM': float(data.get('BankATM', 0.0)),
            'SchoolDistance': float(data.get('SchoolDistance', 0.0)),
            'DistToTown': float(data.get('DistToTown', 0.0)),
            'isfloodedArea': 1 if data.get('isfloodedArea') else 0,
            'IsTsunmaiAlertedArea': 1 if data.get('IsTsunmaiAlertedArea') else 0,
            'PredictionDate': pd.to_datetime(data.get('PredictionDate', datetime.now().strftime("%Y-%m-%d"))),
            'BiddedDate': pd.to_datetime(datetime.now().strftime("%Y-%m-%d")),
            'SoldDate': pd.to_datetime(datetime.now().strftime("%Y-%m-%d"))
        }
        
        # 2. Preprocessing (paraphrased from predict_price.py)
        df = pd.DataFrame([user_data])
        
        # Feature Engineering
        df['DaysOnMarket'] = (df['SoldDate'] - df['BiddedDate']).dt.days
        df['DaysOnMarket'] = df['DaysOnMarket'].fillna(0)
        
        df['Year'] = df['PredictionDate'].dt.year
        df['Month'] = df['PredictionDate'].dt.month
        df['Quarter'] = df['PredictionDate'].dt.quarter
        
        df['RiskScore'] = df['isfloodedArea'] + df['IsTsunmaiAlertedArea']
        
        distances = ['HospitalDistance', 'Bank/ATM', 'SchoolDistance']
        df['InfrastructureScore'] = 1.0 / (df[distances].mean(axis=1) + 0.01)
        
        dataset_dist_max = sample_df['DistToTown'].max() if sample_df is not None else 1.0
        df['EnvironmentalQuality'] = 100 * (1 - df['DistToTown'] / dataset_dist_max)
        df['DistanceToCenter'] = np.sqrt(df['Latitude']**2 + df['Longtitude']**2)
        df['AmenityScore'] = 100 * (1 - df['DistToTown'] / dataset_dist_max)
        
        # Encoding
        le = LabelEncoder()
        if sample_df is not None:
            le.fit(sample_df['city'].unique())
        df['city'] = le.transform(df['city'])
        
        # Feature Selection
        feature_cols = model_info['feature_columns']
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        X = df[feature_cols].copy()
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        # 3. Prediction
        model_name = model_info['model_name']
        scaling_models = ['KNN', 'SVR', 'Linear Regression', 'Ridge Regression', 'Lasso Regression']
        
        if model_name in scaling_models:
            X_input = scaler.transform(X.values)
        else:
            X_input = X.values
            
        prediction = model.predict(X_input)[0]
        prediction = max(prediction, 100000)
        prediction = round(prediction / 1000) * 1000
        
        # Stats
        accuracy = model_info['average_accuracy']
        confidence_margin = prediction * (100 - accuracy) / 100
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'lower_bound': float(prediction - confidence_margin),
            'upper_bound': float(prediction + confidence_margin),
            'accuracy': float(accuracy),
            'model_name': model_name
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
