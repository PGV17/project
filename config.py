"""
Configuration settings for CredTech Intelligence Platform
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = os.getenv('DB_HOST', 'localhost')
    port: int = int(os.getenv('DB_PORT', '5432'))
    database: str = os.getenv('DB_NAME', 'credtech')
    username: str = os.getenv('DB_USER', 'credtech_user')
    password: str = os.getenv('DB_PASSWORD', '')


@dataclass
class APIConfig:
    """External API configuration"""
    alpha_vantage_key: str = os.getenv('ALPHA_VANTAGE_API_KEY', '')
    news_api_key: str = os.getenv('NEWS_API_KEY', '')
    fred_api_key: str = os.getenv('FRED_API_KEY', '')
    yahoo_finance_timeout: int = 30
    cache_duration: int = 300  # 5 minutes


@dataclass
class ModelConfig:
    """Machine learning model configuration"""
    model_path: str = os.getenv('MODEL_PATH', './models/')
    retrain_interval_hours: int = 24
    feature_importance_threshold: float = 0.01
    
    # Credit scoring weights
    positive_factor_weights: Dict[str, float] = None
    negative_factor_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.positive_factor_weights is None:
            self.positive_factor_weights = {
                'roe_score': 0.15,
                'liquidity_score': 0.12,
                'profitability_score': 0.13,
                'growth_score': 0.10,
                'momentum_score': 0.08,
                'sentiment_boost': 0.07
            }
        
        if self.negative_factor_weights is None:
            self.negative_factor_weights = {
                'debt_to_equity_norm': 0.15,
                'volatility_risk': 0.10,
                'macro_risk': 0.08,
                'market_stress': 0.07
            }


@dataclass
class AppConfig:
    """Main application configuration"""
    app_name: str = "CredTech Intelligence Platform"
    version: str = "1.0.0"
    environment: str = os.getenv('ENVIRONMENT', 'development')
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    
    # Streamlit settings
    page_title: str = "CredTech Intelligence Platform"
    page_icon: str = "⚡"
    layout: str = "wide"
    
    # Default companies for analysis
    default_companies: List[str] = None
    
    def __post_init__(self):
        if self.default_companies is None:
            self.default_companies = [
                'AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM', 
                'BAC', 'XOM', 'JNJ', 'PG', 'KO'
            ]


# Global configuration instance
config = AppConfig()
db_config = DatabaseConfig()
api_config = APIConfig()
model_config = ModelConfig()
