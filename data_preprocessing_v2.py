import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class LandValuePreprocessor:
    """
    Data preprocessing for Land Value Prediction
    With DETAILED LOGGING of every step
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.le_dict = {}
        self.scaler = StandardScaler()
        self.original_shape = df.shape
        self.log_step("Initialized preprocessor", f"Dataset: {df.shape[0]} rows × {df.shape[1]} cols")
        
    def log_step(self, title, detail=""):
        """Print formatted log messages"""
        print(f"  📊 {title}")
        if detail:
            print(f"     └─ {detail}")
    
    def log_stats(self, label, series, truncate=2):
        """Print statistics for a series"""
        print(f"     📈 {label}:")
        print(f"        Min: {series.min():>15,.{truncate}f}")
        print(f"        Max: {series.max():>15,.{truncate}f}")
        print(f"        Mean: {series.mean():>14,.{truncate}f}")
        print(f"        Median: {series.median():>12,.{truncate}f}")
        print(f"        Std Dev: {series.std():>11,.{truncate}f}")
    
    def handle_missing_values(self):
        """Handle missing values appropriately with detailed logging"""
        print("\n" + "="*80)
        print("STEP 1: HANDLING MISSING VALUES")
        print("="*80)
        
        # Check initial missing values
        print(f"\n📋 Initial Missing Values Check:")
        missing_count = self.df.isnull().sum().sum()
        if missing_count > 0:
            missing_by_col = self.df.isnull().sum()
            missing_by_col = missing_by_col[missing_by_col > 0]
            print(f"   Total NaN cells: {missing_count}")
            for col, count in missing_by_col.items():
                pct = (count / len(self.df)) * 100
                print(f"   • {col}: {count} missing ({pct:.1f}%)")
        else:
            print(f"   ✓ No missing values found")
        
        # Convert dates
        self.log_step("Converting date columns to datetime format")
        self.df['BiddedDate'] = pd.to_datetime(self.df['BiddedDate'], errors='coerce')
        self.df['SoldDate'] = pd.to_datetime(self.df['SoldDate'], errors='coerce')
        print(f"     ✓ BiddedDate: {self.df['BiddedDate'].dtype}")
        print(f"     ✓ SoldDate: {self.df['SoldDate'].dtype}")
        
        # Fill BiddedDate missing with SoldDate
        self.log_step("Imputing BiddedDate with SoldDate where missing")
        mask = self.df['BiddedDate'].isna()
        fill_count = mask.sum()
        self.df.loc[mask, 'BiddedDate'] = self.df.loc[mask, 'SoldDate']
        print(f"     ✓ Filled {fill_count} missing BiddedDate values")
        
        # Fill numeric columns with median
        self.log_step("Imputing numeric columns with median")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            nan_count = self.df[col].isnull().sum()
            if nan_count > 0:
                median_val = self.df[col].median()
                self.df[col].fillna(median_val, inplace=True)
                print(f"     • {col}: {nan_count} NaN → median {median_val:.2f}")
        
        # Final check
        final_missing = self.df.isnull().sum().sum()
        self.log_step(f"Missing values after imputation: {final_missing}")
        
        return self
    
    def feature_engineering(self):
        """Create new features with detailed logging"""
        print("\n" + "="*80)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*80)
        
        print("\n🏗️  Building Temporal Features:")
        
        # Days on market
        self.log_step("Days on Market (SoldDate - BiddedDate)")
        days_diff = (self.df['SoldDate'] - self.df['BiddedDate']).dt.days
        self.df['DaysOnMarket'] = days_diff.clip(lower=0)
        self.df['DaysOnMarket'].fillna(self.df['DaysOnMarket'].median(), inplace=True)
        self.log_stats("DaysOnMarket (days)", self.df['DaysOnMarket'], truncate=0)
        
        # Year and month of sale
        self.log_step("Extracting Year, Month, Quarter from SoldDate")
        self.df['SaleYear'] = self.df['SoldDate'].dt.year
        self.df['SaleMonth'] = self.df['SoldDate'].dt.month
        self.df['SaleQuarter'] = self.df['SoldDate'].dt.quarter
        print(f"     • Years range: {self.df['SaleYear'].min()} to {self.df['SaleYear'].max()}")
        print(f"     • Months: 1-12 (all months represented)")
        print(f"     • Quarters: Q1-Q4")
        
        print("\n🏗️  Building Risk & Infrastructure Features:")
        
        # Risk score
        self.log_step("Risk Score (Flooding + Tsunami Alert)")
        self.df['RiskScore'] = self.df['isfloodedArea'] + self.df['IsTsunmaiAlertedArea']
        risk_dist = self.df['RiskScore'].value_counts().sort_index()
        print(f"     • No Risk (0): {risk_dist.get(0, 0)} properties")
        print(f"     • Some Risk (1): {risk_dist.get(1, 0)} properties")
        print(f"     • High Risk (2): {risk_dist.get(2, 0)} properties")
        
        # Infrastructure score
        self.log_step("Infrastructure Score (Hospital + School + Bank avg distance)")
        self.df['InfrastructureScore'] = (self.df['HospitalDistance'] + 
                                          self.df['SchoolDistance'] + 
                                          self.df['Bank/ATM']) / 3
        self.log_stats("InfrastructureScore", self.df['InfrastructureScore'])
        
        # Amenity score
        self.log_step("Amenity Score (Tourism + Population avg)")
        self.df['AmenityScore'] = (self.df['TouristScore'] + self.df['Population index']) / 2
        self.log_stats("AmenityScore", self.df['AmenityScore'])
        
        # Environmental quality
        self.log_step("Environmental Quality (100 - Pollution Index)")
        self.df['EnvironmentalQuality'] = 100 - self.df['Pollution index']
        self.log_stats("EnvironmentalQuality", self.df['EnvironmentalQuality'])
        
        # Distance to center
        self.log_step("Distance to Center (√(Latitude² + Longitude²))")
        self.df['DistanceToCenter'] = np.sqrt(self.df['Latitude']**2 + self.df['Longtitude']**2)
        self.log_stats("DistanceToCenter", self.df['DistanceToCenter'])
        
        # Fill any NaN in new features
        feature_cols = ['DaysOnMarket', 'SaleYear', 'SaleMonth', 'SaleQuarter', 'RiskScore',
                       'InfrastructureScore', 'AmenityScore', 'EnvironmentalQuality', 'DistanceToCenter']
        for col in feature_cols:
            self.df[col].fillna(self.df[col].median(), inplace=True)
        
        print(f"\n✅ Feature Engineering Complete - Created 9 new features")
        print(f"   Total features now: {len([c for c in self.df.columns if c != 'Price'])}")
        
        return self
    
    def encode_categorical(self):
        """Encode categorical variables with logging"""
        print("\n" + "="*80)
        print("STEP 3: CATEGORICAL ENCODING")
        print("="*80)
        
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        print(f"\n🔤 Found {len(categorical_cols)} categorical columns:")
        
        for col in categorical_cols:
            if col not in ['SoldDate', 'BiddedDate']:
                unique_vals = self.df[col].nunique()
                print(f"\n   • {col}:")
                print(f"     Unique values: {unique_vals}")
                print(f"     Values: {self.df[col].unique()[:5]}")  # Show first 5
                
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.le_dict[col] = le
                
                print(f"     ✓ Encoded to: {list(range(unique_vals))}")
        
        print(f"\n✅ Encoding Complete")
        
        return self
    
    def prepare_features_and_target(self):
        """Separate features and target with logging"""
        print("\n" + "="*80)
        print("STEP 4: PREPARING FEATURES AND TARGET")
        print("="*80)
        
        # Drop date columns and coordinates (replaced by features)
        drop_cols = ['SoldDate', 'BiddedDate', 'Latitude', 'Longtitude']
        X = self.df.drop(columns=['Price'] + drop_cols)
        y = self.df['Price']
        
        print(f"\n📦 Data Split:")
        print(f"   • Features (X): {X.shape}")
        print(f"   • Target (y): {y.shape}")
        
        # Final NaN check and fill
        initial_nan = X.isnull().sum().sum()
        if initial_nan > 0:
            print(f"\n⚠️  Detected {initial_nan} NaN values - filling with median")
            X = X.fillna(X.median(numeric_only=True))
        
        # Remove rows with any remaining NaN
        valid_rows = ~(X.isna().any(axis=1) | y.isna())
        removed = (~valid_rows).sum()
        
        if removed > 0:
            print(f"   • Removed {removed} rows with unfillable NaN")
        
        X = X[valid_rows].reset_index(drop=True)
        y = y[valid_rows].reset_index(drop=True)
        
        print(f"\n✅ Final Dataset:")
        print(f"   • Samples: {len(X)}")
        print(f"   • Features: {X.shape[1]}")
        print(f"   • Feature names: {list(X.columns)[:5]} ... ({X.shape[1]} total)")
        
        print(f"\n📊 Target Variable Statistics:")
        self.log_stats("Land Price", y)
        
        print(f"\n✅ Preparation Complete")
        
        return X, y, self.scaler
    
    def get_processed_data(self):
        """Complete preprocessing pipeline"""
        self.handle_missing_values()
        self.feature_engineering()
        self.encode_categorical()
        X, y, scaler = self.prepare_features_and_target()
        
        return X, y, scaler


def scale_features_within_cv(X_train, X_test, scaler):
    """
    Scale features without data leakage
    Fit on training data only
    """
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled
