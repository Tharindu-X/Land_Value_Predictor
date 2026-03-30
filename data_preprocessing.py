import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

class LandValuePreprocessor:
    """
    Data preprocessing for Land Value Prediction
    Handles: missing values, feature engineering, encoding, scaling
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.le_dict = {}
        self.scaler = StandardScaler()
        
    def handle_missing_values(self):
        """Handle missing values appropriately"""
        # For BiddedDate, fill with SoldDate if missing
        self.df['BiddedDate'] = pd.to_datetime(self.df['BiddedDate'], errors='coerce')
        self.df['SoldDate'] = pd.to_datetime(self.df['SoldDate'], errors='coerce')
        
        # Fill BiddedDate missing with SoldDate
        mask = self.df['BiddedDate'].isna()
        self.df.loc[mask, 'BiddedDate'] = self.df.loc[mask, 'SoldDate']
        
        # Fill any remaining NaN values with median (for numeric columns)
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].median(), inplace=True)
        
        # Fill remaining NaN values in object columns with mode
        object_cols = self.df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if self.df[col].isnull().sum() > 0:
                self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown', inplace=True)
        
        print(f"✓ Missing values handled")
        return self
    
    def feature_engineering(self):
        """Create new features from existing data"""
        # Days on market (time between bid and sale)
        days_diff = (self.df['SoldDate'] - self.df['BiddedDate']).dt.days
        self.df['DaysOnMarket'] = days_diff.clip(lower=0)  # No negative values
        self.df['DaysOnMarket'].fillna(self.df['DaysOnMarket'].median(), inplace=True)
        
        # Year and month of sale
        self.df['SaleYear'] = self.df['SoldDate'].dt.year
        self.df['SaleMonth'] = self.df['SoldDate'].dt.month
        self.df['SaleQuarter'] = self.df['SoldDate'].dt.quarter
        
        # Fill any NaN created from date extraction
        self.df['SaleYear'].fillna(self.df['SaleYear'].median(), inplace=True)
        self.df['SaleMonth'].fillna(self.df['SaleMonth'].median(), inplace=True)
        self.df['SaleQuarter'].fillna(self.df['SaleQuarter'].median(), inplace=True)
        
        # Risk score (combination of flood and tsunami risk)
        self.df['RiskScore'] = self.df['isfloodedArea'] + self.df['IsTsunmaiAlertedArea']
        
        # Infrastructure score (proximity to essentials)
        self.df['InfrastructureScore'] = (self.df['HospitalDistance'] + 
                                           self.df['SchoolDistance'] + 
                                           self.df['Bank/ATM']) / 3
        
        # Amenity score (what makes area attractive)
        self.df['AmenityScore'] = (self.df['TouristScore'] + self.df['Population index']) / 2
        
        # Environmental quality score
        self.df['EnvironmentalQuality'] = 100 - self.df['Pollution index']
        
        # Distance to center (using Manhattan distance approximation)
        self.df['DistanceToCenter'] = np.sqrt(self.df['Latitude']**2 + self.df['Longtitude']**2)
        
        # Fill any NaN in new features
        numeric_cols = ['DaysOnMarket', 'SaleYear', 'SaleMonth', 'SaleQuarter', 'RiskScore',
                       'InfrastructureScore', 'AmenityScore', 'EnvironmentalQuality', 'DistanceToCenter']
        for col in numeric_cols:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        
        print(f"✓ Feature engineering completed - {self.df.shape[1]} features now")
        return self
    
    def encode_categorical(self):
        """Encode categorical variables"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['SoldDate', 'BiddedDate']:  # Skip date columns
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.le_dict[col] = le
        
        print(f"✓ Categorical encoding completed")
        return self
    
    def prepare_features_and_target(self):
        """Separate features and target, drop date columns"""
        # Drop original date columns and geographic coordinates (replaced by features)
        drop_cols = ['SoldDate', 'BiddedDate', 'Latitude', 'Longtitude']
        X = self.df.drop(columns=['Price'] + drop_cols)
        y = self.df['Price']
        
        # Final pass - fill any remaining NaN values
        X = X.fillna(X.median(numeric_only=True))
        
        # Drop any rows with NaN that couldn't be filled
        valid_rows = ~(X.isna().any(axis=1) | y.isna())
        X = X[valid_rows].reset_index(drop=True)
        y = y[valid_rows].reset_index(drop=True)
        
        print(f"✓ Features and target separated")
        print(f"  X shape: {X.shape}")
        print(f"  y shape: {y.shape}")
        print(f"  Features: {list(X.columns)}")
        
        return X, y
    
    def get_processed_data(self):
        """Complete preprocessing pipeline"""
        self.handle_missing_values()
        self.feature_engineering()
        self.encode_categorical()
        X, y = self.prepare_features_and_target()
        
        return X, y, self.scaler


def create_time_series_splits(X, y, n_splits=5):
    """
    Create time-based splits for temporal models (LSTM, ARIMA)
    Training on earlier dates, testing on later dates
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    
    for train_idx, test_idx in tscv.split(X):
        splits.append((train_idx, test_idx))
    
    return splits


def scale_features_within_cv(X_train, X_test, scaler):
    """
    CRITICAL: Scale features without data leakage
    - Fit scaler ONLY on training data
    - Apply to both train and test
    """
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
