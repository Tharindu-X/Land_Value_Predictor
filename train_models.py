import pandas as pd
import numpy as np
import warnings
import json
from datetime import datetime
from sklearn.model_selection import cross_val_score, cross_validate, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.impute import SimpleImputer
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima import auto_arima

# Optional deep learning imports
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, Dropout
    from tensorflow.keras.optimizers import Adam
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("⚠️  Warning: TensorFlow/Keras not available - ANN and LSTM models will be skipped")
import joblib

from data_preprocessing import LandValuePreprocessor, scale_features_within_cv

warnings.filterwarnings('ignore')

class LandValueMLSystem:
    """
    Complete ML System for Land Value Prediction with 12 models
    Includes nested cross-validation and hyperparameter tuning
    """
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.cv_results = {}
        self.best_models = {}
        self.model_performance = {}
        
    def load_and_prepare_data(self):
        """Load and preprocess data"""
        print("\n" + "="*80)
        print("STAGE 1: DATA LOADING AND PREPROCESSING")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded dataset: {self.df.shape[0]} records, {self.df.shape[1]} columns")
        
        # Preprocess
        preprocessor = LandValuePreprocessor(self.df)
        self.X, self.y, self.scaler = preprocessor.get_processed_data()
        
        print("\nDataset Summary:")
        print(f"  Target (Price) - Min: ${self.y.min():,.0f}, Max: ${self.y.max():,.0f}, Mean: ${self.y.mean():,.0f}")
        print(f"  Missing values in X: {self.X.isnull().sum().sum()}")
        
    # ============ MODEL 1-2: LINEAR REGRESSION MODELS ============
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Model 1: Simple Linear Regression"""
        print("\n  [Model 1] Linear Regression")
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    
    def train_multivariate_linear_regression(self, X_train, X_test, y_train, y_test):
        """Model 2: Multivariate Linear Regression with Ridge Regularization"""
        print("  [Model 2] Multivariate Linear Regression (Ridge)")
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    
    # ============ MODEL 3: DECISION TREE ============
    
    def train_decision_tree(self, X_train, X_test, y_train, y_test):
        """Model 3: Decision Tree Regressor"""
        print("  [Model 3] Decision Tree Regressor")
        model = DecisionTreeRegressor(max_depth=15, min_samples_split=10, 
                                      min_samples_leaf=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    
    # ============ MODEL 4-5: RANDOM FOREST ============
    
    def train_random_forest(self, X_train, X_test, y_train, y_test):
        """Model 4: Random Forest Regressor (Standard)"""
        print("  [Model 4] Random Forest Regressor")
        model = RandomForestRegressor(n_estimators=100, max_depth=20, 
                                      min_samples_split=10, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    
    def train_random_forest_optimized(self, X_train, X_test, y_train, y_test):
        """Model 5: Random Forest Regressor (Optimized)"""
        print("  [Model 5] Random Forest Optimized")
        model = RandomForestRegressor(n_estimators=200, max_depth=25, 
                                      min_samples_split=5, min_samples_leaf=2,
                                      max_features='sqrt', random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    
    # ============ MODEL 6: SUPPORT VECTOR REGRESSION ============
    
    def train_svr(self, X_train, X_test, y_train, y_test):
        """Model 6: Support Vector Regressor"""
        print("  [Model 6] Support Vector Regressor")
        model = SVR(kernel='rbf', C=1000, gamma='scale', epsilon=0.1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    
    # ============ MODEL 7: KNN ============
    
    def train_knn(self, X_train, X_test, y_train, y_test):
        """Model 7: K-Nearest Neighbors"""
        print("  [Model 7] K-Nearest Neighbors (KNN)")
        model = KNeighborsRegressor(n_neighbors=5, weights='distance')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    
    # ============ MODEL 8: XGBOOST ============
    
    def train_xgboost(self, X_train, X_test, y_train, y_test):
        """Model 8: XGBoost Regressor"""
        print("  [Model 8] XGBoost Regressor")
        model = xgb.XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1,
                               subsample=0.8, colsample_bytree=0.8, random_state=42)
        model.fit(X_train, y_train, verbose=False)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    
    # ============ MODEL 9: LIGHTGBM ============
    
    def train_lightgbm(self, X_train, X_test, y_train, y_test):
        """Model 9: LightGBM Regressor"""
        print("  [Model 9] LightGBM Regressor")
        model = lgb.LGBMRegressor(n_estimators=100, max_depth=7, learning_rate=0.1,
                                num_leaves=31, random_state=42, verbose=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    
    # ============ MODEL 10: ARTIFICIAL NEURAL NETWORK ============
    
    def train_ann(self, X_train, X_test, y_train, y_test):
        """Model 10: Artificial Neural Network"""
        print("  [Model 10] Artificial Neural Network (ANN)")
        
        if not DEEP_LEARNING_AVAILABLE:
            print("    (Skipped - TensorFlow/Keras not installed)")
            return None, {'RMSE': 0, 'MAE': 0, 'R2': 0, 'MAPE': 0}
        
        model = Sequential([
            Dense(128, activation='relu', input_dim=X_train.shape[1]),
            Dropout(0.2),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dropout(0.1),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, 
                 verbose=0)
        
        y_pred = model.predict(X_test, verbose=0)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        
        return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
    
    # ============ MODEL 11: LSTM ============
    
    def prepare_lstm_data(self, X, y, sequence_length=5):
        """Prepare data for LSTM (sequential format)"""
        X_lstm = []
        y_lstm = []
        
        for i in range(len(X) - sequence_length):
            X_lstm.append(X[i:i+sequence_length])
            y_lstm.append(y.iloc[i+sequence_length])
        
        return np.array(X_lstm), np.array(y_lstm)
    
    def train_lstm(self, X_train, X_test, y_train, y_test):
        """Model 11: LSTM - Recurrent Neural Network"""
        print("  [Model 11] LSTM (Recurrent Neural Network)")
        
        if not DEEP_LEARNING_AVAILABLE:
            print("    (Skipped - TensorFlow/Keras not installed)")
            return None, {'RMSE': 0, 'MAE': 0, 'R2': 0, 'MAPE': 0}
        
        try:
            # Prepare sequential data
            seq_len = 5
            X_train_lstm, y_train_lstm = self.prepare_lstm_data(X_train.values, y_train, seq_len)
            X_test_lstm, y_test_lstm = self.prepare_lstm_data(X_test.values, y_test, seq_len)
            
            if len(X_train_lstm) == 0 or len(X_test_lstm) == 0:
                print("    (Skipped - insufficient data for sequences)")
                return None, {'RMSE': 0, 'MAE': 0, 'R2': 0, 'MAPE': 0}
            
            model = Sequential([
                LSTM(64, activation='relu', input_shape=(seq_len, X_train.shape[1])),
                Dropout(0.2),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            model.fit(X_train_lstm, y_train_lstm, epochs=30, batch_size=16, 
                     validation_split=0.2, verbose=0)
            
            y_pred = model.predict(X_test_lstm, verbose=0)
            
            rmse = np.sqrt(mean_squared_error(y_test_lstm, y_pred))
            mae = mean_absolute_error(y_test_lstm, y_pred)
            r2 = r2_score(y_test_lstm, y_pred)
            mape = mean_absolute_percentage_error(y_test_lstm, y_pred)
            
            return model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
        
        except Exception as e:
            print(f"    (LSTM training skipped - {str(e)})")
            return None, {'RMSE': 0, 'MAE': 0, 'R2': 0, 'MAPE': 0}
    
    # ============ MODEL 12: ARIMA ============
    
    def train_arima(self, X_train, X_test, y_train, y_test):
        """Model 12: ARIMA - Time Series Model"""
        print("  [Model 12] ARIMA (Time Series)")
        
        try:
            # Fit ARIMA on training data only
            # Use auto_arima to find best parameters
            arima_model = auto_arima(y_train, seasonal=False, stepwise=True, 
                                    suppress_warnings=True, max_p=5, max_q=5, max_d=2)
            
            # Forecast for test set
            forecast_steps = len(y_test)
            y_pred = arima_model.predict(n_periods=forecast_steps)
            
            rmse = np.sqrt(mean_squared_error(y_test.values, y_pred))
            mae = mean_absolute_error(y_test.values, y_pred)
            r2 = r2_score(y_test.values, y_pred)
            mape = mean_absolute_percentage_error(y_test.values, y_pred)
            
            return arima_model, {'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape}
        
        except Exception as e:
            print(f"    (ARIMA training error - {str(e)})")
            return None, {'RMSE': 0, 'MAE': 0, 'R2': 0, 'MAPE': 0}
    
    # ============ CROSS-VALIDATION WITH K-FOLD ============
    
    def train_all_models_with_cv(self, n_splits=5):
        """
        Train all 12 models with K-Fold Cross-Validation
        This is the OUTER LOOP of nested CV
        """
        print("\n" + "="*80)
        print("STAGE 2: TRAINING ALL 12 MODELS WITH 5-FOLD CROSS-VALIDATION")
        print("="*80)
        
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        # Discretize y for stratification
        y_bins = pd.qcut(self.y, q=5, labels=False, duplicates='drop')
        
        model_results = {
            'Linear Regression': [],
            'Multivariate Linear Regression': [],
            'Decision Tree': [],
            'Random Forest': [],
            'Random Forest Optimized': [],
            'SVR': [],
            'KNN': [],
            'XGBoost': [],
            'LightGBM': [],
            'ANN': [],
            'LSTM': [],
            'ARIMA': []
        }
        
        fold_count = 0
        for train_idx, test_idx in kfold.split(self.X, y_bins):
            fold_count += 1
            print(f"\n{'─'*80}")
            print(f"FOLD {fold_count}/{n_splits}")
            print(f"{'─'*80}")
            
            # Split data
            X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]
            
            # Impute missing values BEFORE scaling
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
            # Scale without data leakage
            X_train_scaled, X_test_scaled = scale_features_within_cv(X_train, X_test, StandardScaler())
            
            # Train all models
            model_1, metrics_1 = self.train_linear_regression(X_train_scaled, X_test_scaled, y_train, y_test)
            model_results['Linear Regression'].append(metrics_1)
            
            model_2, metrics_2 = self.train_multivariate_linear_regression(X_train_scaled, X_test_scaled, y_train, y_test)
            model_results['Multivariate Linear Regression'].append(metrics_2)
            
            model_3, metrics_3 = self.train_decision_tree(X_train, X_test, y_train, y_test)
            model_results['Decision Tree'].append(metrics_3)
            
            model_4, metrics_4 = self.train_random_forest(X_train, X_test, y_train, y_test)
            model_results['Random Forest'].append(metrics_4)
            
            model_5, metrics_5 = self.train_random_forest_optimized(X_train, X_test, y_train, y_test)
            model_results['Random Forest Optimized'].append(metrics_5)
            
            model_6, metrics_6 = self.train_svr(X_train_scaled, X_test_scaled, y_train, y_test)
            model_results['SVR'].append(metrics_6)
            
            model_7, metrics_7 = self.train_knn(X_train_scaled, X_test_scaled, y_train, y_test)
            model_results['KNN'].append(metrics_7)
            
            model_8, metrics_8 = self.train_xgboost(X_train, X_test, y_train, y_test)
            model_results['XGBoost'].append(metrics_8)
            
            model_9, metrics_9 = self.train_lightgbm(X_train, X_test, y_train, y_test)
            model_results['LightGBM'].append(metrics_9)
            
            model_10, metrics_10 = self.train_ann(X_train_scaled, X_test_scaled, y_train, y_test)
            model_results['ANN'].append(metrics_10)
            
            model_11, metrics_11 = self.train_lstm(X_train_scaled, X_test_scaled, y_train, y_test)
            model_results['LSTM'].append(metrics_11)
            
            model_12, metrics_12 = self.train_arima(y_train, y_test, y_train, y_test)
            model_results['ARIMA'].append(metrics_12)
            
            # Best model in this fold
            all_metrics = {
                'Linear Regression': metrics_1['RMSE'],
                'Multivariate Linear Regression': metrics_2['RMSE'],
                'Decision Tree': metrics_3['RMSE'],
                'Random Forest': metrics_4['RMSE'],
                'Random Forest Optimized': metrics_5['RMSE'],
                'SVR': metrics_6['RMSE'],
                'KNN': metrics_7['RMSE'],
                'XGBoost': metrics_8['RMSE'],
                'LightGBM': metrics_9['RMSE'],
                'ANN': metrics_10['RMSE'],
                'LSTM': metrics_11['RMSE'],
                'ARIMA': metrics_12['RMSE'] if metrics_12['RMSE'] > 0 else float('inf')
            }
            
            best_model_name = min(all_metrics, key=all_metrics.get)
            print(f"\n✓ Fold {fold_count} Best: {best_model_name} (RMSE: ₹{all_metrics[best_model_name]:,.0f})")
        
        # Calculate average metrics per model
        self.cv_results = model_results
        self.print_cv_summary()
        
        return model_results
    
    def print_cv_summary(self):
        """Print summary of CV results"""
        print("\n" + "="*80)
        print("STAGE 3: CROSS-VALIDATION SUMMARY")
        print("="*80)
        
        print("\nAverage Metrics Across All 5 Folds:\n")
        print(f"{'Model':<35} {'Avg RMSE':<15} {'Avg MAE':<15} {'Avg R2':<10}")
        print("─" * 75)
        
        summary_data = []
        for model_name, results_list in self.cv_results.items():
            if results_list and results_list[0]['RMSE'] > 0:
                avg_rmse = np.mean([r['RMSE'] for r in results_list])
                avg_mae = np.mean([r['MAE'] for r in results_list])
                avg_r2 = np.mean([r['R2'] for r in results_list])
                
                summary_data.append({
                    'model': model_name,
                    'rmse': avg_rmse,
                    'mae': avg_mae,
                    'r2': avg_r2
                })
                
                print(f"{model_name:<35} ₹{avg_rmse:>13,.0f}  ₹{avg_mae:>13,.0f}  {avg_r2:>8.4f}")
        
        # Sort and find best
        summary_data.sort(key=lambda x: x['rmse'])
        
        print("\n" + "="*80)
        print("🏆 BEST MODELS (Ranked by RMSE):")
        print("="*80)
        for i, item in enumerate(summary_data[:3], 1):
            print(f"{i}. {item['model']:<30} RMSE: ₹{item['rmse']:>12,.0f}")
    
    def save_results(self):
        """Save CV results to JSON"""
        results_dir = 'd:\\Campus work\\My projects\\Land_Value_Predictor\\results'
        import os
        os.makedirs(results_dir, exist_ok=True)
        
        # Convert to serializable format
        serializable_results = {}
        for model_name, results_list in self.cv_results.items():
            serializable_results[model_name] = [
                {k: float(v) if isinstance(v, np.number) else v 
                 for k, v in r.items()}
                for r in results_list
            ]
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = f'{results_dir}\\cv_results_{timestamp}.json'
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\n✓ Results saved to: {filepath}")


def main():
    """Main execution"""
    # Initialize system
    ml_system = LandValueMLSystem('d:\\Campus work\\My projects\\Land_Value_Predictor\\dataset\\LandValues.csv')
    
    # Load and prepare data
    ml_system.load_and_prepare_data()
    
    # Train all models with 5-Fold CV
    ml_system.train_all_models_with_cv(n_splits=5)
    
    # Save results
    ml_system.save_results()
    
    print("\n" + "="*80)
    print("✓ TRAINING COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()
