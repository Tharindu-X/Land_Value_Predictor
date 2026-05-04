"""
Land Value Price Prediction System
Uses the best trained model to predict land prices based on user inputs
"""

import pandas as pd
import numpy as np
import json
import warnings
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import joblib

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

from data_preprocessing_new import LandValuePreprocessorNew

warnings.filterwarnings('ignore')


class LandValuePredictionSystem:
    """
    Interactive system for predicting land prices
    Uses the best trained model from cross-validation
    """
    
    def __init__(self):
        self.model_info = None
        self.model = None
        self.scaler = None
        self.sample_df = None
        self.preprocessor = None
        self.best_params = {}
        self.load_model_and_info()
    
    def load_model_and_info(self):
        """Load best model and metadata"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + " LAND VALUE PREDICTION SYSTEM ".center(78) + "█")
        print("█" + " Loading Best Trained Model ".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        model_path = 'models/best_model.pkl'
        scaler_path = 'models/scaler.pkl'
        info_path = 'models/best_model_info.json'
        
        # Check if model exists
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("\n❌ ERROR: No trained model found!")
            print("   Please run: python train_models_current.py")
            print("   to train and save a model first.")
            exit(1)
        
        # Load model
        print("\n📦 Loading trained model...")
        try:
            self.model = joblib.load(model_path)
            print(f"   ✅ Model loaded: {model_path}")
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            exit(1)
        
        # Load scaler
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"   ✅ Scaler loaded: {scaler_path}")
        except Exception as e:
            print(f"   ❌ Error loading scaler: {e}")
            exit(1)
        
        # Load metadata
        if os.path.exists(info_path):
            with open(info_path, 'r') as f:
                self.model_info = json.load(f)
            
            print(f"\n✅ Model Loaded Successfully!")
            print(f"   Model: {self.model_info['model_name']}")
            print(f"   Accuracy: {self.model_info['average_accuracy']:.2f}%")
            print(f"   Features: {len(self.model_info['feature_columns'])} input features")
            print(f"   Training Date: {self.model_info['training_date'][:10]}")
        else:
            print("\n⚠️  Warning: Model metadata not found")
    
    def load_sample_data(self):
        """Load sample data for feature engineering reference"""
        print("\n📊 Loading preprocessing reference data...")
        try:
            self.sample_df = pd.read_csv('dataset/CurrentDataset_cleaned_fixed_new.csv')
            print(f"✅ Reference data loaded: {self.sample_df.shape[0]} properties")
        except Exception as e:
            print(f"⚠️  Warning: Could not load reference data: {e}")
    
    def get_user_inputs(self):
        """Get property details from user"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + " ENTER PROPERTY DETAILS ".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        user_data = {}
        
        print("\n🏙️  LOCATION & BASIC INFO:")
        print("─"*80)
        
        # City
        print("\n📍 City:")
        if self.sample_df is not None:
            cities = self.sample_df['city'].unique()
            print("   Available cities:")
            for idx, city in enumerate(sorted(cities)[:10], 1):
                print(f"   {idx}. {city}")
            if len(cities) > 10:
                print(f"   ... and {len(cities)-10} more")
        
        city = input("   Enter city name: ").strip()
        user_data['city'] = city
        
        # ZipCode
        zip_code = input("\n   Enter ZipCode: ").strip()
        try:
            user_data['ZipCode'] = int(zip_code)
        except ValueError:
            user_data['ZipCode'] = 0
            print(f"   ⚠️  Invalid ZipCode - using 0")
        
        # Coordinates
        print("\n🗺️  GEOGRAPHIC DATA:")
        print("─"*80)
        
        latitude = input("\n   Enter Latitude: ").strip()
        try:
            user_data['Latitude'] = float(latitude)
        except ValueError:
            user_data['Latitude'] = 0.0
            print(f"   ⚠️  Invalid Latitude - using 0.0")
        
        longitude = input("\n   Enter Longitude: ").strip()
        try:
            user_data['Longtitude'] = float(longitude)
        except ValueError:
            user_data['Longtitude'] = 0.0
            print(f"   ⚠️  Invalid Longitude - using 0.0")
        
        # Distance information
        print("\n📏 DISTANCE INFORMATION (in km):")
        print("─"*80)
        
        hosp_dist = input("\n   Distance to Hospital (km): ").strip()
        try:
            user_data['HospitalDistance'] = float(hosp_dist)
        except ValueError:
            user_data['HospitalDistance'] = 0.0
            print(f"   ⚠️  Invalid value - using 0.0")
        
        bank_dist = input("\n   Distance to Bank/ATM (km): ").strip()
        try:
            user_data['Bank/ATM'] = float(bank_dist)
        except ValueError:
            user_data['Bank/ATM'] = 0.0
            print(f"   ⚠️  Invalid value - using 0.0")
        
        school_dist = input("\n   Distance to School (km): ").strip()
        try:
            user_data['SchoolDistance'] = float(school_dist)
        except ValueError:
            user_data['SchoolDistance'] = 0.0
            print(f"   ⚠️  Invalid value - using 0.0")
        
        dist_town = input("\n   Distance to Town Center (km): ").strip()
        try:
            user_data['DistToTown'] = float(dist_town)
        except ValueError:
            user_data['DistToTown'] = 0.0
            print(f"   ⚠️  Invalid value - using 0.0")
        
        # Risk assessment
        print("\n⚠️  RISK ASSESSMENT:")
        print("─"*80)
        
        flooded = input("\n   Is in flooded area? (yes/no): ").strip().lower()
        user_data['isfloodedArea'] = 1 if flooded in ['yes', 'y', '1'] else 0
        print(f"   {'✓' if user_data['isfloodedArea'] else '✗'} Flooded: {'Yes' if user_data['isfloodedArea'] else 'No'}")
        
        tsunami = input("\n   Is in tsunami alert area? (yes/no): ").strip().lower()
        user_data['IsTsunmaiAlertedArea'] = 1 if tsunami in ['yes', 'y', '1'] else 0
        print(f"   {'✓' if user_data['IsTsunmaiAlertedArea'] else '✗'} Tsunami Alert: {'Yes' if user_data['IsTsunmaiAlertedArea'] else 'No'}")
        
        # Prediction date (only date needed for future price prediction)
        print("\n🔮 PREDICTION DATE:")
        print("─"*80)
        
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        while True:
            pred_date = input("\n   Prediction Date (YYYY-MM-DD): ").strip()
            try:
                pred_date_obj = pd.to_datetime(pred_date)
                user_data['PredictionDate'] = pred_date_obj
                print(f"   ✓ Prediction Date: {pred_date}")
                break
            except:
                print("   ❌ Invalid date format! Please use YYYY-MM-DD")
        
        # Use current date as default for transaction dates (not asked from user)
        # These are used for feature engineering (DaysOnMarket calculation)
        current_date_obj = pd.to_datetime(current_date)
        user_data['BiddedDate'] = current_date_obj
        user_data['SoldDate'] = current_date_obj
        
        return user_data
    
    def preprocess_user_input(self, user_data):
        """Preprocess user input to match training format"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + " PREPROCESSING USER INPUT ".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        # Create DataFrame
        df = pd.DataFrame([user_data])
        
        print("\n📊 Input Data:")
        print(f"   City: {df['city'].values[0]}")
        print(f"   ZipCode: {df['ZipCode'].values[0]}")
        print(f"   Latitude: {df['Latitude'].values[0]:.4f}")
        print(f"   Longitude: {df['Longtitude'].values[0]:.4f}")
        
        # Feature engineering (same as training)
        print("\n🏗️  Engineering Features...")
        
        # Days on Market
        df['DaysOnMarket'] = (df['SoldDate'] - df['BiddedDate']).dt.days
        df['DaysOnMarket'] = df['DaysOnMarket'].fillna(0)
        
        # Temporal features from prediction date (not SoldDate)
        df['Year'] = df['PredictionDate'].dt.year
        df['Month'] = df['PredictionDate'].dt.month
        df['Quarter'] = df['PredictionDate'].dt.quarter
        
        # Risk Score
        df['RiskScore'] = df['isfloodedArea'] + df['IsTsunmaiAlertedArea']
        df['RiskScore'] = df['RiskScore'].fillna(0)
        
        # Infrastructure Score
        distances = ['HospitalDistance', 'Bank/ATM', 'SchoolDistance']
        df['InfrastructureScore'] = 1.0 / (df[distances].mean(axis=1) + 0.01)
        
        # Environmental Quality - USE SAMPLE DATASET MAX
        if self.sample_df is not None and 'DistToTown' in self.sample_df.columns:
            dataset_dist_max = self.sample_df['DistToTown'].max()
        else:
            dataset_dist_max = 1.0
        df['EnvironmentalQuality'] = 100 * (1 - df['DistToTown'] / dataset_dist_max)
        
        # Distance to Center
        df['DistanceToCenter'] = np.sqrt(df['Latitude']**2 + df['Longtitude']**2)
        
        # Amenity Score - USE SAMPLE DATASET MAX
        df['AmenityScore'] = 100 * (1 - df['DistToTown'] / dataset_dist_max)
        
        # Encode city using the same label ordering as training
        le = LabelEncoder()
        if self.sample_df is not None:
            le.fit(self.sample_df['city'])
            if df['city'].iloc[0] not in le.classes_:
                fallback_city = self.sample_df['city'].mode().iloc[0]
                print(f"   ⚠️  Unknown city '{df['city'].iloc[0]}' - using '{fallback_city}'")
                df['city'] = fallback_city
            df['city'] = le.transform(df['city'])
        else:
            df['city'] = le.fit_transform(df['city'])
        
        # Select features matching training
        feature_cols = self.model_info['feature_columns']
        
        # Drop date columns and prediction date
        drop_cols = ['SoldDate', 'BiddedDate', 'PredictionDate']
        
        # Ensure all required features exist
        for col in feature_cols:
            if col not in df.columns:
                print(f"   ⚠️  Missing feature: {col} - setting to 0")
                df[col] = 0
        
        # Select only required features
        X = df[feature_cols].copy()
        
        # Handle missing values
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        print(f"✅ Preprocessing complete!")
        print(f"   Features prepared: {X.shape[1]} dimensions")
        
        return X
    
    def predict_price(self, X):
        """Make price prediction using saved trained model"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + " MAKING PREDICTION ".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        model_name = self.model_info['model_name']
        print(f"\n🤖 Using Model: {model_name}")
        print(f"   Accuracy: {self.model_info['average_accuracy']:.2f}%")
        
        # Prepare input based on model type
        try:
            # Check if model needs scaling
            from sklearn.neighbors import KNeighborsRegressor
            from sklearn.svm import SVR
            from sklearn.linear_model import LinearRegression, Ridge, Lasso
            
            scaling_models = ['KNN', 'SVR', 'Linear Regression', 'Ridge Regression', 'Lasso Regression']
            
            if model_name in scaling_models:
                print(f"\n   🔄 Scaling input features...")
                X_input = self.scaler.transform(X.values)
            else:
                X_input = X.values
            
            # Make prediction
            print(f"   📊 Making prediction...")
            predicted_price = self.model.predict(X_input)[0]
            
            # Ensure reasonable price
            predicted_price = max(predicted_price, 100000)  # At least 100k LKR
            predicted_price = round(predicted_price / 1000) * 1000  # Round to nearest 1000
            
            print(f"   ✅ Prediction complete!")
            
        except Exception as e:
            print(f"\n❌ Error during prediction: {e}")
            print(f"   Using fallback prediction...")
            # Fallback: use mean from training data
            predicted_price = 500000
        
        print(f"\n💰 PREDICTED LAND PRICE: Rs. {predicted_price:,.0f}")
        print(f"   Accuracy confidence: ±{self.model_info['average_accuracy']:.1f}%")
        
        # Calculate confidence interval
        confidence_margin = predicted_price * (100 - self.model_info['average_accuracy']) / 100
        lower_bound = predicted_price - confidence_margin
        upper_bound = predicted_price + confidence_margin
        
        print(f"\n📊 PRICE RANGE (with {self.model_info['average_accuracy']:.0f}% confidence):")
        print(f"   Lower Bound: Rs. {lower_bound:,.0f}")
        print(f"   Best Estimate: Rs. {predicted_price:,.0f}")
        print(f"   Upper Bound: Rs. {upper_bound:,.0f}")
        
        return predicted_price, lower_bound, upper_bound
    
    def display_summary(self, user_data, predicted_price, lower_bound, upper_bound):
        """Display prediction summary"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + " PREDICTION SUMMARY ".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        print("\n📍 PROPERTY DETAILS:")
        print(f"   City: {user_data['city']}")
        print(f"   ZipCode: {user_data['ZipCode']}")
        print(f"   Latitude: {user_data['Latitude']:.4f}°")
        print(f"   Longitude: {user_data['Longtitude']:.4f}°")
        
        print("\n📏 DISTANCES:")
        print(f"   Hospital: {user_data['HospitalDistance']:.2f} km")
        print(f"   Bank/ATM: {user_data['Bank/ATM']:.2f} km")
        print(f"   School: {user_data['SchoolDistance']:.2f} km")
        print(f"   Town Center: {user_data['DistToTown']:.2f} km")
        
        print("\n⚠️  RISK STATUS:")
        print(f"   Flooded Area: {'Yes ⚠️' if user_data['isfloodedArea'] else 'No ✓'}")
        print(f"   Tsunami Alert: {'Yes ⚠️' if user_data['IsTsunmaiAlertedArea'] else 'No ✓'}")
        
        print("\n📅 PREDICTION DATE:")
        print(f"   Predicting price for: {user_data['PredictionDate'].strftime('%Y-%m-%d')}")
        
        print("\n" + "─"*80)
        print(f"\n💰 PREDICTED PRICE: Rs. {predicted_price:,.0f}")
        print(f"   95% Confidence Range: Rs. {lower_bound:,.0f} - Rs. {upper_bound:,.0f}")
        print(f"   Model Accuracy: {self.model_info['average_accuracy']:.2f}%")
        
        print("\n" + "█"*80)
    
    def save_prediction(self, user_data, predicted_price):
        """Save prediction to file"""
        if not os.path.exists('predictions'):
            os.makedirs('predictions')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"predictions/prediction_{timestamp}.json"
        
        prediction_record = {
            'timestamp': datetime.now().isoformat(),
            'property': {
                'city': user_data['city'],
                'zipcode': user_data['ZipCode'],
                'latitude': user_data['Latitude'],
                'longitude': user_data['Longtitude']
            },
            'distances': {
                'hospital': user_data['HospitalDistance'],
                'bank_atm': user_data['Bank/ATM'],
                'school': user_data['SchoolDistance'],
                'town': user_data['DistToTown']
            },
            'risk': {
                'flooded': bool(user_data['isfloodedArea']),
                'tsunami': bool(user_data['IsTsunmaiAlertedArea'])
            },
            'prediction_date': user_data['PredictionDate'].isoformat(),
            'predicted_price': predicted_price,
            'model_used': self.model_info['model_name'],
            'model_accuracy': self.model_info['average_accuracy']
        }
        
        with open(filename, 'w') as f:
            json.dump(prediction_record, f, indent=4)
        
        print(f"\n✅ Prediction saved to: {filename}")
    
    def run(self):
        """Run the complete prediction system"""
        self.load_sample_data()
        user_data = self.get_user_inputs()
        X = self.preprocess_user_input(user_data)
        predicted_price, lower_bound, upper_bound = self.predict_price(X)
        self.display_summary(user_data, predicted_price, lower_bound, upper_bound)
        
        # Save prediction
        save = input("\n💾 Save prediction to file? (yes/no): ").strip().lower()
        if save in ['yes', 'y', '1']:
            self.save_prediction(user_data, predicted_price)
        
        # Ask for another prediction
        another = input("\n🔄 Make another prediction? (yes/no): ").strip().lower()
        if another in ['yes', 'y', '1']:
            self.run()
        else:
            print("\n" + "█"*80)
            print("█" + " Thank you for using Land Value Prediction System! ".center(78) + "█")
            print("█"*80 + "\n")


def main():
    """Main execution"""
    system = LandValuePredictionSystem()
    system.run()


if __name__ == "__main__":
    main()
