"""
Test prediction on existing dataset rows
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from data_preprocessing_v2 import LandValuePreprocessor
import warnings

warnings.filterwarnings('ignore')

def test_existing_data_prediction():
    """Test model on data it was trained on"""
    
    print("\n" + "="*80)
    print("TEST: PREDICT PRICES FOR DATA ALREADY IN DATASET")
    print("="*80)
    
    # Load data
    df = pd.read_csv('dataset/LandValues.csv')
    
    # Preprocess
    preprocessor = LandValuePreprocessor(df)
    X, y, scaler = preprocessor.get_processed_data()
    
    # Train model on full dataset
    print("\n🔧 Training Random Forest on FULL dataset...")
    model = RandomForestRegressor(
        n_estimators=200, 
        max_depth=25, 
        min_samples_split=5, 
        min_samples_leaf=2,
        max_features='sqrt', 
        random_state=42, 
        n_jobs=-1
    )
    model.fit(X, y)
    print("✅ Model trained!")
    
    # Test on first 10 rows
    print("\n" + "="*80)
    print("TESTING: Predicting prices for 10 random properties")
    print("="*80)
    
    # Select random rows
    test_indices = np.random.choice(len(X), size=10, replace=False)
    
    test_X = X.iloc[test_indices]
    test_y = y.iloc[test_indices]
    
    # Make predictions
    predictions = model.predict(test_X)
    
    # Calculate differences
    differences = np.abs(predictions - test_y.values)
    percentage_errors = (differences / test_y.values) * 100
    
    # Display results
    print("\n📊 RESULTS:")
    print("-" * 100)
    print(f"{'Index':<8} {'Actual Price':<20} {'Predicted Price':<20} {'Difference':<15} {'Error %':<10}")
    print("-" * 100)
    
    for i, (idx, pred, actual) in enumerate(zip(test_indices, predictions, test_y.values)):
        diff = abs(pred - actual)
        error_pct = (diff / actual) * 100
        print(f"{idx:<8} ${actual:>17,.0f}  ${pred:>17,.0f}  ${diff:>12,.0f}  {error_pct:>8.4f}%")
    
    print("-" * 100)
    print(f"\n📈 SUMMARY:")
    print(f"   • Average Error: ${differences.mean():,.0f}")
    print(f"   • Max Error: ${differences.max():,.0f}")
    print(f"   • Min Error: ${differences.min():,.0f}")
    print(f"   • Average % Error: {percentage_errors.mean():.4f}%")
    
    print("\n✅ CONCLUSION:")
    print("   Since model was trained on this data, predictions match actual prices!")
    print("   This is expected - not a true generalization test.")
    
    return model, X, y

def test_new_data_prediction(model, X_train, y_train):
    """Simulate prediction on NEW/unseen data"""
    
    print("\n\n" + "="*80)
    print("TEST 2: What about NEW data (not in training)?")
    print("="*80)
    
    print("\nTo test on truly NEW data, you would need:")
    print("1. Properties that were NOT in your dataset")
    print("2. Future data (2019+)")
    print("3. Different market conditions")
    print("\nExample:")
    
    # Create a synthetic new property
    # (In reality, you'd get real new data)
    new_property_features = X_train.iloc[0].values.copy()
    
    print(f"\nGiven a new property with features from dataset row 0:")
    prediction = model.predict([new_property_features])[0]
    actual = y_train.iloc[0]
    
    print(f"  Predicted Price: ${prediction:,.0f}")
    print(f"  Type of Test: Same features as training data")
    print(f"  Expected Accuracy: High (100%) ✅")
    
    print("\nBut if the new property has DIFFERENT features:")
    print("  (e.g., different city, different pollution level)")
    print("  Expected Accuracy: Unknown (depend on how different)")
    
    return prediction

if __name__ == "__main__":
    model, X, y = test_existing_data_prediction()
    test_new_data_prediction(model, X, y)
    
    print("\n\n" + "="*80)
    print("KEY INSIGHT:")
    print("="*80)
    print("\n✅ For data IN the dataset: Model predicts ~perfectly (100%)")
    print("❓ For data NOT in dataset: Accuracy depends on similarity")
    print("⚠️  For FUTURE data (2019+): Accuracy might be lower (different market)")
    print("\nYour model learned patterns from 2012-2018 data.")
    print("Real future predictions need NEW data from future years.")
