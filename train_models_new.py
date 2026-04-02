"""
Train Models on NEW DATASET (Columns Removed: Pollution Index, Tourist Score, Population Index)
"""

import pandas as pd
import numpy as np
import warnings
import json
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                            mean_absolute_percentage_error, median_absolute_error)
import xgboost as xgb
import lightgbm as lgb
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
import joblib

from data_preprocessing_new import LandValuePreprocessorNew, scale_features_within_cv

warnings.filterwarnings('ignore')

class ImprovedLandValueMLSystemNew:
    """
    Enhanced ML System for NEW DATASET (without Pollution Index, Tourist Score, Population Index)
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
        """Load and preprocess data with detailed output"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + " LAND VALUE PREDICTION ML SYSTEM - NEW DATASET ".center(78) + "█")
        print("█" + " (Removed: Pollution Index, Tourist Score, Population Index) ".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        print("\n" + "="*80)
        print("LOADING DATASET")
        print("="*80)
        
        self.df = pd.read_csv(self.data_path)
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   📁 File: {self.data_path}")
        print(f"   📊 Dimensions: {self.df.shape[0]:,} records × {self.df.shape[1]} columns")
        
        # Preprocess
        print("\n" + "="*80)
        print("DATA PREPROCESSING")
        print("="*80)
        
        preprocessor = LandValuePreprocessorNew(self.df)
        self.X, self.y, self.scaler = preprocessor.get_processed_data()
        
        print("\n" + "="*80)
        print("PREPROCESSING SUMMARY")
        print("="*80)
        print(f"\n📊 Final Dataset Ready for Training:")
        print(f"   • Features (X): {self.X.shape}")
        print(f"   • Target (y): {self.y.shape}")
        print(f"   • Price Range: ${self.y.min():,.0f} to ${self.y.max():,.0f}")
        print(f"   • Average Price: ${self.y.mean():,.0f}")
        print(f"   • Features Ready: ✅ No missing values")
        
    def calculate_advanced_metrics(self, y_true, y_pred, model_name):
        """Calculate comprehensive accuracy metrics"""
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        median_ae = median_absolute_error(y_true, y_pred)
        
        # Accuracy percentage (100 - MAPE)
        accuracy_pct = 100 - mape
        
        # Mean Absolute Percentage Error ranges
        mape_ranges = {
            'Excellent (< 5%)': mape < 5,
            'Very Good (5-10%)': 5 <= mape < 10,
            'Good (10-20%)': 10 <= mape < 20,
            'Fair (20-50%)': 20 <= mape < 50,
            'Poor (> 50%)': mape >= 50
        }
        
        accuracy_rating = [k for k, v in mape_ranges.items() if v][0]
        
        return {
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Accuracy%': accuracy_pct,
            'MedianAE': median_ae,
            'Rating': accuracy_rating
        }
    
    def train_all_models_with_cv(self, n_splits=5):
        """Train all models with detailed K-Fold Cross-Validation"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + " CROSS-VALIDATION MODEL TRAINING ".center(78) + "█")
        print("█" + f" Using {n_splits}-Fold Stratified Cross-Validation ".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_bins = pd.qcut(self.y, q=5, labels=False, duplicates='drop')
        
        model_results = {
            'Linear Regression': [],
            'Ridge Regression': [],
            'Lasso Regression': [],
            'Decision Tree': [],
            'Random Forest': [],
            'Random Forest Optimized': [],
            'Gradient Boosting': [],
            'SVR': [],
            'KNN': [],
            'XGBoost': [],
            'LightGBM': [],
            'ARIMA': []
        }
        
        fold_count = 0
        for train_idx, test_idx in kfold.split(self.X, y_bins):
            fold_count += 1
            print(f"\n{'█'*80}")
            print(f"█ FOLD {fold_count}/{n_splits} ".ljust(79) + "█")
            print(f"{'█'*80}")
            
            # Split data
            X_train, X_test = self.X.iloc[train_idx].values, self.X.iloc[test_idx].values
            y_train, y_test = self.y.iloc[train_idx].values, self.y.iloc[test_idx].values
            
            print(f"\n📊 Fold {fold_count} Data Distribution:")
            print(f"   Train set: {X_train.shape[0]} samples")
            print(f"   Test set: {X_test.shape[0]} samples")
            print(f"   Features: {X_train.shape[1]}")
            
            # Scale without data leakage
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
            
            X_train_scaled, X_test_scaled = scale_features_within_cv(X_train, X_test, StandardScaler())
            
            # Train models
            print(f"\n🤖 Training Models on Fold {fold_count}:")
            
            # Model 1: Linear Regression
            print(f"\n   [1/12] Linear Regression...", end=" ")
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            y_pred = lr.predict(X_test_scaled)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "Linear Regression")
            model_results['Linear Regression'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 2: Ridge Regression
            print(f"   [2/12] Ridge Regression...", end=" ")
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_train_scaled, y_train)
            y_pred = ridge.predict(X_test_scaled)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "Ridge Regression")
            model_results['Ridge Regression'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 3: Lasso Regression
            print(f"   [3/12] Lasso Regression...", end=" ")
            lasso = Lasso(alpha=1.0)
            lasso.fit(X_train_scaled, y_train)
            y_pred = lasso.predict(X_test_scaled)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "Lasso Regression")
            model_results['Lasso Regression'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 4: Decision Tree
            print(f"   [4/12] Decision Tree...", end=" ")
            dt = DecisionTreeRegressor(max_depth=15, min_samples_split=10, random_state=42)
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "Decision Tree")
            model_results['Decision Tree'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 5: Random Forest
            print(f"   [5/12] Random Forest...", end=" ")
            rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "Random Forest")
            model_results['Random Forest'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 6: Random Forest Optimized
            print(f"   [6/12] Random Forest Optimized...", end=" ")
            rf_opt = RandomForestRegressor(n_estimators=200, max_depth=25, min_samples_split=5,
                                          max_features='sqrt', random_state=42, n_jobs=-1)
            rf_opt.fit(X_train, y_train)
            y_pred = rf_opt.predict(X_test)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "Random Forest Optimized")
            model_results['Random Forest Optimized'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 7: Gradient Boosting
            print(f"   [7/12] Gradient Boosting...", end=" ")
            gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            gb.fit(X_train, y_train)
            y_pred = gb.predict(X_test)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "Gradient Boosting")
            model_results['Gradient Boosting'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 8: SVR
            print(f"   [8/12] Support Vector Regressor...", end=" ")
            svr = SVR(kernel='rbf', C=1000, gamma='scale')
            svr.fit(X_train_scaled, y_train)
            y_pred = svr.predict(X_test_scaled)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "SVR")
            model_results['SVR'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 9: KNN
            print(f"   [9/12] K-Nearest Neighbors...", end=" ")
            knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
            knn.fit(X_train_scaled, y_train)
            y_pred = knn.predict(X_test_scaled)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "KNN")
            model_results['KNN'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 10: XGBoost
            print(f"   [10/12] XGBoost...", end=" ")
            xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42)
            xgb_model.fit(X_train, y_train, verbose=False)
            y_pred = xgb_model.predict(X_test)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "XGBoost")
            model_results['XGBoost'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 11: LightGBM
            print(f"   [11/12] LightGBM...", end=" ")
            lgb_model = lgb.LGBMRegressor(n_estimators=100, max_depth=7, learning_rate=0.1, random_state=42, verbose=-1)
            lgb_model.fit(X_train, y_train)
            y_pred = lgb_model.predict(X_test)
            metrics = self.calculate_advanced_metrics(y_test, y_pred, "LightGBM")
            model_results['LightGBM'].append(metrics)
            print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            
            # Model 12: ARIMA
            print(f"   [12/12] ARIMA...", end=" ")
            try:
                arima_model = auto_arima(y_train, seasonal=False, stepwise=True, suppress_warnings=True,
                                        max_p=5, max_q=5, max_d=2)
                y_pred = arima_model.predict(n_periods=len(y_test))
                metrics = self.calculate_advanced_metrics(y_test, y_pred, "ARIMA")
                model_results['ARIMA'].append(metrics)
                print(f"✅ (Acc: {metrics['Accuracy%']:.2f}%)")
            except Exception as e:
                print(f"⚠️  (Error: {str(e)[:30]})")
                model_results['ARIMA'].append({'RMSE': 0, 'MAE': 0, 'R2': 0, 'MAPE': 100, 'Accuracy%': 0, 'MedianAE': 0, 'Rating': 'Error'})
            
            # Find best model in fold
            best_fold_model = min([(name, results[-1]['RMSE']) for name, results in model_results.items() 
                                  if results[-1]['RMSE'] > 0], key=lambda x: x[1])[0]
            
            print(f"\n🏆 Fold {fold_count} Best: {best_fold_model} (Accuracy: {model_results[best_fold_model][-1]['Accuracy%']:.2f}%)")
        
        self.cv_results = model_results
        self.print_cv_summary()
        
        return model_results
    
    def print_cv_summary(self):
        """Print detailed CV summary with accuracy metrics"""
        print("\n" + "█"*80)
        print("█" + " "*78 + "█")
        print("█" + " CROSS-VALIDATION RESULTS & ACCURACY ".center(78) + "█")
        print("█" + " "*78 + "█")
        print("█"*80)
        
        print("\n📊 AVERAGE METRICS ACROSS ALL FOLDS:\n")
        print(f"{'Model':<30} {'Accuracy%':>10} {'RMSE':>15} {'R2 Score':>10} {'Rating':<15}")
        print("─"*80)
        
        summary_data = []
        for model_name, results_list in self.cv_results.items():
            if results_list and results_list[0]['RMSE'] > 0:
                avg_acc = np.mean([r['Accuracy%'] for r in results_list])
                avg_rmse = np.mean([r['RMSE'] for r in results_list])
                avg_r2 = np.mean([r['R2'] for r in results_list])
                avg_rating = results_list[0]['Rating']
                
                summary_data.append({
                    'model': model_name,
                    'accuracy': avg_acc,
                    'rmse': avg_rmse,
                    'r2': avg_r2,
                    'rating': avg_rating
                })
                
                print(f"{model_name:<30} {avg_acc:>9.2f}% ${avg_rmse:>13,.0f}  {avg_r2:>9.4f}  {avg_rating:<15}")
        
        # Find top 5
        summary_data.sort(key=lambda x: x['accuracy'], reverse=True)
        
        print("\n" + "█"*80)
        print("\n🏆 TOP 5 BEST MODELS (By Accuracy)\n")
        
        medals = ['🥇', '🥈', '🥉', '4️⃣ ', '5️⃣ ']
        for i, item in enumerate(summary_data[:5]):
            medal = medals[i] if i < len(medals) else f"{i+1}. "
            print(f"{medal} {i+1}. {item['model']:<25} Accuracy: {item['accuracy']:>6.2f}%  RMSE: ${item['rmse']:>12,.0f}")
        
        print("\n" + "█"*80)
        print("\n✅ Training Complete!\n")


def main():
    """Main execution"""
    system = ImprovedLandValueMLSystemNew('dataset/LandSale_removescolumns.csv')
    system.load_and_prepare_data()
    system.train_all_models_with_cv(n_splits=5)
    print("\n✅ Pipeline execution completed successfully!")


if __name__ == "__main__":
    main()
