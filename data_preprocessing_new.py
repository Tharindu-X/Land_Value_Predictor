"""
Data Preprocessing for Land Values - NEW DATASET (Removed 3 columns)
Removed: Pollution index, Tourist score, Population index
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer


class LandValuePreprocessorNew:
    """
    Preprocessing pipeline for new dataset (without Pollution Index, Tourist Score, Population Index)
    Handles missing values, feature engineering, categorical encoding, and normalization
    """
    
    def __init__(self, df):
        self.df = df.copy()
        self.scaler = StandardScaler()
        
        print("\n" + "📊 Initialized preprocessor")
        print(f"   └─ Dataset: {self.df.shape[0]} rows × {self.df.shape[1]} cols")
        
    def handle_missing_values(self):
        """Handle missing values with strategic imputation"""
        print("\n" + "="*80)
        print("STEP 1: HANDLING MISSING VALUES")
        print("="*80)
        
        # Check missing values
        missing_info = self.df.isnull().sum()
        missing_pct = (missing_info / len(self.df)) * 100
        
        print(f"\n📋 Initial Missing Values Check:")
        print(f"   Total NaN cells: {missing_info.sum()}")
        
        for col in missing_info[missing_info > 0].index:
            print(f"   • {col}: {missing_info[col]} missing ({missing_pct[col]:.1f}%)")
        
        # Convert date columns
        print(f"\n  📊 Converting date columns to datetime format")
        for col in ['BiddedDate', 'SoldDate']:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                print(f"     ✓ {col}: datetime64[us]")
        
        # Impute BiddedDate with SoldDate where missing
        if 'BiddedDate' in self.df.columns and 'SoldDate' in self.df.columns:
            missing_bid = self.df['BiddedDate'].isnull().sum()
            self.df['BiddedDate'] = self.df['BiddedDate'].fillna(self.df['SoldDate'])
            print(f"  📊 Imputing BiddedDate with SoldDate where missing")
            print(f"     ✓ Filled {missing_bid} missing BiddedDate values")
        
        # Impute numeric columns with median
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        cols_with_missing = missing_info[missing_info > 0].index
        if len(cols_with_missing) > 0:
            print(f"  📊 Imputing numeric columns with median")
            for col in cols_with_missing:
                if col in numeric_cols:
                    median_val = self.df[col].median()
                    print(f"     • {col}: {missing_info[col]} NaN → median {median_val:.2f}")
                    self.df[col] = self.df[col].fillna(median_val)
        
        # Final check
        remaining_nan = self.df.isnull().sum().sum()
        print(f"  📊 Missing values after imputation: {remaining_nan}")
    
    def feature_engineering(self):
        """Create new features from existing data"""
        print("\n" + "="*80)
        print("STEP 2: FEATURE ENGINEERING")
        print("="*80)
        
        # Temporal features
        print(f"\n🏗️  Building Temporal Features:")
        
        # Days on Market
        self.df['DaysOnMarket'] = (self.df['SoldDate'] - self.df['BiddedDate']).dt.days
        self.df['DaysOnMarket'] = self.df['DaysOnMarket'].fillna(0)
        
        print(f"  📊 Days on Market (SoldDate - BiddedDate)")
        print(f"     📈 DaysOnMarket (days):")
        print(f"        Min:   {self.df['DaysOnMarket'].min():>15,.0f}")
        print(f"        Max:   {self.df['DaysOnMarket'].max():>15,.0f}")
        print(f"        Mean:  {self.df['DaysOnMarket'].mean():>15,.0f}")
        print(f"        Median:{self.df['DaysOnMarket'].median():>15,.0f}")
        print(f"        Std Dev:{self.df['DaysOnMarket'].std():>14,.0f}")
        
        # Temporal features from SoldDate
        self.df['Year'] = self.df['SoldDate'].dt.year
        self.df['Month'] = self.df['SoldDate'].dt.month
        self.df['Quarter'] = self.df['SoldDate'].dt.quarter
        
        print(f"  📊 Extracting Year, Month, Quarter from SoldDate")
        print(f"     • Years range: {self.df['Year'].min()} to {self.df['Year'].max()}")
        print(f"     • Months: {self.df['Month'].min()}-{self.df['Month'].max()} (all months represented)")
        print(f"     • Quarters: Q1-Q{self.df['Quarter'].max()}")
        
        # Risk Score (Flooding + Tsunami)
        print(f"\n🏗️  Building Risk & Infrastructure Features:")
        self.df['RiskScore'] = self.df['isfloodedArea'] + self.df['IsTsunmaiAlertedArea']
        self.df['RiskScore'] = self.df['RiskScore'].fillna(0)  # Fill any NaN values
        
        print(f"  📊 Risk Score (Flooding + Tsunami Alert)")
        for risk_level in sorted(self.df['RiskScore'].unique()):
            if not pd.isna(risk_level):  # Only process non-NaN values
                count = (self.df['RiskScore'] == risk_level).sum()
                print(f"     • Risk Level {int(risk_level)}: {count} properties")
        
        # Infrastructure Score (Average of hospital, bank, school distances)
        # Lower distance = better = higher score (so we invert)
        # Using harmonic mean for distances (better for rates)
        distances = ['HospitalDistance', 'Bank/ATM', 'SchoolDistance']
        self.df['InfrastructureScore'] = 1.0 / (self.df[distances].mean(axis=1) + 0.01)
        
        print(f"  📊 Infrastructure Score (Hospital + Bank + School avg distance)")
        print(f"     📈 InfrastructureScore:")
        print(f"        Min:    {self.df['InfrastructureScore'].min():>14,.2f}")
        print(f"        Max:    {self.df['InfrastructureScore'].max():>14,.2f}")
        print(f"        Mean:   {self.df['InfrastructureScore'].mean():>14,.2f}")
        print(f"        Median: {self.df['InfrastructureScore'].median():>14,.2f}")
        print(f"        Std Dev:{self.df['InfrastructureScore'].std():>14,.2f}")
        
        # Environmental Quality (NOTE: No pollution index in new dataset)
        # Using DistToTown as proxy for environmental quality
        self.df['EnvironmentalQuality'] = 100 * (1 - self.df['DistToTown'] / self.df['DistToTown'].max())
        
        print(f"  📊 Environmental Quality (Distance to Town based)⚠️ (Pollution Index removed)")
        print(f"     📈 EnvironmentalQuality:")
        print(f"        Min:    {self.df['EnvironmentalQuality'].min():>14,.2f}")
        print(f"        Max:    {self.df['EnvironmentalQuality'].max():>14,.2f}")
        print(f"        Mean:   {self.df['EnvironmentalQuality'].mean():>14,.2f}")
        print(f"        Median: {self.df['EnvironmentalQuality'].median():>14,.2f}")
        print(f"        Std Dev:{self.df['EnvironmentalQuality'].std():>14,.2f}")
        
        # Distance to Center
        self.df['DistanceToCenter'] = np.sqrt(self.df['Latitude']**2 + self.df['Longtitude']**2)
        
        print(f"  📊 Distance to Center (√(Latitude² + Longitude²))")
        print(f"     📈 DistanceToCenter:")
        print(f"        Min:    {self.df['DistanceToCenter'].min():>14,.2f}")
        print(f"        Max:    {self.df['DistanceToCenter'].max():>14,.2f}")
        print(f"        Mean:   {self.df['DistanceToCenter'].mean():>14,.2f}")
        print(f"        Median: {self.df['DistanceToCenter'].median():>14,.2f}")
        print(f"        Std Dev:{self.df['DistanceToCenter'].std():>14,.2f}")
        
        # Amenity Score (NOTE: No Tourist Score or Population Index in new dataset)
        # Using DistToTown as proxy for amenities
        self.df['AmenityScore'] = 100 * (1 - self.df['DistToTown'] / self.df['DistToTown'].max())
        
        print(f"  📊 Amenity Score (Distance to Town based)⚠️ (Tourist & Population removed)")
        print(f"     📈 AmenityScore:")
        print(f"        Min:    {self.df['AmenityScore'].min():>14,.2f}")
        print(f"        Max:    {self.df['AmenityScore'].max():>14,.2f}")
        print(f"        Mean:   {self.df['AmenityScore'].mean():>14,.2f}")
        print(f"        Median: {self.df['AmenityScore'].median():>14,.2f}")
        print(f"        Std Dev:{self.df['AmenityScore'].std():>14,.2f}")
        
        print(f"\n✅ Feature Engineering Complete - Created 8 new features")
        print(f"   Total features now: {self.df.shape[1]}")
    
    def encode_categorical(self):
        """Encode categorical variables"""
        print("\n" + "="*80)
        print("STEP 3: CATEGORICAL ENCODING")
        print("="*80)
        
        # Find categorical columns (both 'object' and 'string' types)
        categorical_cols = self.df.select_dtypes(include=['object', 'string']).columns
        print(f"\n🔤 Found {len(categorical_cols)} categorical columns:\n")
        
        le_dict = {}
        for col in categorical_cols:
            unique_vals = self.df[col].unique()
            print(f"   • {col}:")
            print(f"     Unique values: {len(unique_vals)}")
            print(f"     Values: {unique_vals[:5]}...")
            
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            le_dict[col] = le
            print(f"     ✓ Encoded to: {sorted(self.df[col].unique())}")
        
        print(f"\n✅ Encoding Complete")
    
    def prepare_features_and_target(self):
        """Prepare final features and target variable"""
        print("\n" + "="*80)
        print("STEP 4: PREPARING FEATURES AND TARGET")
        print("="*80)
        
        # Drop date columns and target name
        drop_cols = ['SoldDate', 'BiddedDate', 'Price']
        X_cols = [col for col in self.df.columns if col not in drop_cols]
        
        X = self.df[X_cols].copy()
        y = self.df['Price'].copy()
        
        print(f"\n📦 Data Split:")
        print(f"   • Features (X): {X.shape}")
        print(f"   • Target (y): {y.shape}")
        
        # Handle any remaining NaN
        remaining_nan = X.isnull().sum().sum()
        if remaining_nan > 0:
            print(f"\n⚠️  Detected {remaining_nan} NaN values - filling with median")
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        print(f"\n✅ Final Dataset:")
        print(f"   • Samples: {len(X)}")
        print(f"   • Features: {X.shape[1]}")
        print(f"   • Feature names: {list(X.columns[:5])} ... ({X.shape[1]} total)")
        
        print(f"\n📊 Target Variable Statistics:")
        print(f"     📈 Land Price:")
        print(f"        Min:    ${y.min():>17,.2f}")
        print(f"        Max:    ${y.max():>17,.2f}")
        print(f"        Mean:   ${y.mean():>17,.2f}")
        print(f"        Median: ${y.median():>17,.2f}")
        print(f"        Std Dev:${y.std():>17,.2f}")
        
        print(f"\n✅ Preparation Complete")
        
        return X, y
    
    def get_processed_data(self):
        """Execute full preprocessing pipeline"""
        self.handle_missing_values()
        self.feature_engineering()
        self.encode_categorical()
        X, y = self.prepare_features_and_target()
        return X, y, self.scaler


def scale_features_within_cv(X_train, X_test, scaler):
    """Scale features without data leakage"""
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


if __name__ == "__main__":
    # Test the preprocessor
    df = pd.read_csv('dataset/LandSale_removescolumns.csv')
    preprocessor = LandValuePreprocessorNew(df)
    X, y, scaler = preprocessor.get_processed_data()
    print("\n✅ Preprocessing pipeline completed successfully!")
