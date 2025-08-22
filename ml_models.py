"""
Machine Learning Models for Credit Scoring
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from datetime import datetime, timedelta
import joblib
import shap
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Feature engineering for credit scoring"""
    
    def __init__(self):
        self.feature_scalers = {}
        self.feature_history = {}
        
    def create_technical_indicators(self, price_data):
        """Create technical analysis features"""
        if len(price_data) < 20:
            return {}
            
        features = {}
        
        # Moving averages
        features['sma_5'] = price_data[-5:].mean()
        features['sma_10'] = price_data[-10:].mean()
        features['sma_20'] = price_data[-20:].mean()
        
        # Exponential moving averages
        features['ema_12'] = self._calculate_ema(price_data, 12)
        features['ema_26'] = self._calculate_ema(price_data, 26)
        
        # MACD
        macd_line = features['ema_12'] - features['ema_26']
        signal_line = self._calculate_ema([macd_line], 9)
        features['macd'] = macd_line
        features['macd_signal'] = signal_line
        features['macd_histogram'] = macd_line - signal_line
        
        # RSI
        features['rsi'] = self._calculate_rsi(price_data)
        
        # Bollinger Bands
        bb_middle = features['sma_20']
        bb_std = np.std(price_data[-20:])
        features['bb_upper'] = bb_middle + (2 * bb_std)
        features['bb_lower'] = bb_middle - (2 * bb_std)
        features['bb_position'] = (price_data[-1] - bb_middle) / (2 * bb_std)
        
        # Price momentum
        features['momentum_5'] = (price_data[-1] / price_data[-6] - 1) if len(price_data) > 5 else 0
        features['momentum_10'] = (price_data[-1] / price_data[-11] - 1) if len(price_data) > 10 else 0
        
        return features
    
    def _calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        if len(data) < period:
            return np.mean(data)
        
        multiplier = 2 / (period + 1)
        ema = data[0]
        for price in data[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
        return ema
    
    def _calculate_rsi(self, data, period=14):
        """Calculate Relative Strength Index"""
        if len(data) < period + 1:
            return 50
            
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_fundamental_features(self, financial_data):
        """Create fundamental analysis features"""
        features = {}
        
        features['gross_margin'] = financial_data.get('grossMargins', 0)
        features['operating_margin'] = financial_data.get('operatingMargins', 0)
        features['net_margin'] = financial_data.get('profitMargins', 0)
        features['roe'] = financial_data.get('returnOnEquity', 0)
        features['roa'] = financial_data.get('returnOnAssets', 0)
        features['roic'] = financial_data.get('returnOnCapital', 0)
        
        features['current_ratio'] = financial_data.get('currentRatio', 1.0)
        features['quick_ratio'] = financial_data.get('quickRatio', 1.0)
        features['cash_ratio'] = financial_data.get('cashRatio', 0.1)
        
        features['debt_to_equity'] = financial_data.get('debtToEquity', 0)
        features['debt_to_assets'] = financial_data.get('totalDebt', 0) / max(financial_data.get('totalAssets', 1), 1)
        features['interest_coverage'] = financial_data.get('interestCoverage', 1)
        
        features['asset_turnover'] = financial_data.get('assetTurnover', 0)
        features['inventory_turnover'] = financial_data.get('inventoryTurnover', 0)
        features['receivables_turnover'] = financial_data.get('receivablesTurnover', 0)
        
        features['revenue_growth'] = financial_data.get('revenueGrowth', 0)
        features['earnings_growth'] = financial_data.get('earningsGrowth', 0)
        features['book_value_growth'] = financial_data.get('bookValue', 0)
        
        # Valuation metrics
        features['pe_ratio'] = financial_data.get('trailingPE', 15)
        features['pb_ratio'] = financial_data.get('priceToBook', 1)
        features['ps_ratio'] = financial_data.get('priceToSalesTrailing12Months', 1)
        features['peg_ratio'] = financial_data.get('pegRatio', 1)
        
        return features
    
    def create_macro_features(self, macro_data, sector=None):
        """Create macroeconomic features"""
        features = {}
        
        features['interest_rate_level'] = macro_data.get('interest_rate', 5.0)
        features['yield_curve_slope'] = macro_data.get('yield_10y', 4.5) - macro_data.get('yield_2y', 4.0)
        features['credit_spread'] = macro_data.get('credit_spread', 1.5)
        
        features['gdp_growth'] = macro_data.get('gdp_growth', 2.5)
        features['inflation_rate'] = macro_data.get('inflation_rate', 3.0)
        features['unemployment_rate'] = macro_data.get('unemployment_rate', 4.0)
        features['consumer_confidence'] = macro_data.get('consumer_confidence', 100)
        
        features['vix_level'] = macro_data.get('vix_index', 18)
        features['dollar_strength'] = macro_data.get('dollar_index', 103)
        features['commodity_pressure'] = macro_data.get('commodity_index', 100)
        
        if sector:
            features[f'{sector}_sector_beta'] = macro_data.get(f'{sector}_beta', 1.0)
            features[f'{sector}_sector_momentum'] = macro_data.get(f'{sector}_momentum', 0.0)
        
        return features
    
    def create_sentiment_features(self, sentiment_data):
        """Create sentiment and alternative data features"""
        features = {}
        
        features['news_sentiment'] = sentiment_data.get('sentiment_score', 0)
        features['news_volume'] = min(sentiment_data.get('news_volume', 5) / 20.0, 1.0)
        features['sentiment_volatility'] = sentiment_data.get('sentiment_volatility', 0.1)
        
        features['social_mentions'] = sentiment_data.get('social_mentions', 0)
        features['social_sentiment'] = sentiment_data.get('social_sentiment', 0)
        
        features['analyst_upgrades'] = sentiment_data.get('upgrades', 0)
        features['analyst_downgrades'] = sentiment_data.get('downgrades', 0)
        features['consensus_rating'] = sentiment_data.get('consensus_rating', 3.0)
        
        features['esg_score'] = sentiment_data.get('esg_score', 50)
        features['governance_score'] = sentiment_data.get('governance_score', 50)
        
        return features

class ModelEnsemble:
    """Ensemble of multiple ML models for credit scoring"""
    
    def __init__(self):
        self.models = {}
        self.model_weights = {}
        self.feature_scalers = {}
        self.is_trained = False
        
    def initialize_models(self):
        """Initialize multiple model architectures"""
        self.models = {
            'random_forest': RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ),
            'lightgbm': lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            ),
            'neural_network': MLPRegressor(
                hidden_layer_sizes=(128, 64, 32),
                activation='relu',
                solver='adam',
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            ),
            'elastic_net': ElasticNet(
                alpha=0.1,
                l1_ratio=0.5,
                random_state=42
            )
        }
        
        # Initialize scalers for each model
        for model_name in self.models.keys():
            self.feature_scalers[model_name] = StandardScaler()
    
    def train_ensemble(self, X, y, validation_split=0.2):
        """Train all models in the ensemble"""
        if not self.models:
            self.initialize_models()
        
        # Split data for validation
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        model_scores = {}
        
        for model_name, model in self.models.items():
            
            X_train_scaled = self.feature_scalers[model_name].fit_transform(X_train)
            X_val_scaled = self.feature_scalers[model_name].transform(X_val)
            
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_val_scaled)
            score = r2_score(y_val, y_pred)
            model_scores[model_name] = max(score, 0.01)
        
        total_score = sum(model_scores.values())
        self.model_weights = {name: score/total_score for name, score in model_scores.items()}
        
        self.is_trained = True
        return model_scores
    
    def predict(self, X):
        """Make ensemble predictions"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        predictions = {}
        
        for model_name, model in self.models.items():
            X_scaled = self.feature_scalers[model_name].transform(X)
            pred = model.predict(X_scaled)
            predictions[model_name] = pred
        
        ensemble_pred = np.zeros(len(X))
        for model_name, pred in predictions.items():
            ensemble_pred += pred * self.model_weights[model_name]
        
        return ensemble_pred, predictions
    
    def get_feature_importance(self, feature_names):
        """Get aggregated feature importance across models"""
        importance_dict = {}
        
        for model_name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[model_name] = model.feature_importances_
        
        # Average importance across tree-based models
        if importance_dict:
            avg_importance = np.mean(list(importance_dict.values()), axis=0)
            return dict(zip(feature_names, avg_importance))
        
        return {}
    
    def explain_prediction(self, X, feature_names):
        """Generate SHAP explanations for predictions"""
        explanations = {}
        
        # Use the best performing model for SHAP explanation
        best_model_name = max(self.model_weights.items(), key=lambda x: x[1])[0]
        best_model = self.models[best_model_name]
        
        X_scaled = self.feature_scalers[best_model_name].transform(X)
        
        if hasattr(best_model, 'predict'):
            try:
                if 'forest' in best_model_name or 'xgb' in best_model_name or 'lightgbm' in best_model_name:
                    explainer = shap.TreeExplainer(best_model)
                else:
                    explainer = shap.LinearExplainer(best_model, X_scaled)
                
                shap_values = explainer.shap_values(X_scaled)
                
                # Convert to feature contributions
                for i, feature in enumerate(feature_names):
                    explanations[feature] = np.mean(shap_values[:, i]) if len(shap_values.shape) > 1 else shap_values[i]
            
            except Exception as e:
                print(f"SHAP explanation failed: {e}")
                # Fallback to feature importance
                explanations = self.get_feature_importance(feature_names)
        
        return explanations
    
    def save_models(self, filepath_prefix):
        """Save trained models"""
        for model_name, model in self.models.items():
            joblib.dump(model, f"{filepath_prefix}_{model_name}.pkl")
            joblib.dump(self.feature_scalers[model_name], f"{filepath_prefix}_{model_name}_scaler.pkl")
        
        # Save weights and metadata
        joblib.dump({
            'weights': self.model_weights,
            'is_trained': self.is_trained
        }, f"{filepath_prefix}_metadata.pkl")
    
    def load_models(self, filepath_prefix):
        """Load trained models"""
        metadata = joblib.load(f"{filepath_prefix}_metadata.pkl")
        self.model_weights = metadata['weights']
        self.is_trained = metadata['is_trained']
        
        for model_name in self.model_weights.keys():
            self.models[model_name] = joblib.load(f"{filepath_prefix}_{model_name}.pkl")
            self.feature_scalers[model_name] = joblib.load(f"{filepath_prefix}_{model_name}_scaler.pkl")

class RealTimeModelUpdater:
    """Handles incremental learning and model updates"""
    
    def __init__(self, model_ensemble):
        self.ensemble = model_ensemble
        self.update_buffer = []
        self.update_threshold = 50  # Update after 50 new samples
        
    def add_new_data(self, X_new, y_new):
        """Add new data to update buffer"""
        self.update_buffer.append((X_new, y_new))
        
        if len(self.update_buffer) >= self.update_threshold:
            self.incremental_update()
    
    def incremental_update(self):
        """Perform incremental model update"""
        if not self.update_buffer:
            return
        
        # Combine buffered data
        X_updates = np.vstack([x for x, y in self.update_buffer])
        y_updates = np.hstack([y for x, y in self.update_buffer])
        
        # Update models that support incremental learning
        for model_name, model in self.ensemble.models.items():
            if hasattr(model, 'partial_fit'):
                X_scaled = self.ensemble.feature_scalers[model_name].transform(X_updates)
                model.partial_fit(X_scaled, y_updates)
        
        # Clear buffer
        self.update_buffer = []
        
        print(f"Models updated with {len(y_updates)} new samples")
    
    def retrain_schedule(self, schedule_hours=24):
        """Schedule full model retraining"""
        # This would be implemented with a job scheduler in production
        pass