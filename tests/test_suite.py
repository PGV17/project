"""
Test Suite for Financial Analysis Application
"""

import unittest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import DataIngestionEngine, CreditScoringEngine, ExplainabilityEngine
from ml_models import AdvancedFeatureEngineer, ModelEnsemble
from nlp_engine import (
    FinancialEventExtractor, SentimentAnalyzer, NewsAggregator
)

class TestDataIngestionEngine(unittest.TestCase):
    """Test data ingestion and processing"""
    
    def setUp(self):
        self.engine = DataIngestionEngine()
    
    def test_fetch_financial_data_success(self):
        """Test successful financial data fetching"""
        with patch('yfinance.Ticker') as mock_ticker:
            # Mock yfinance response
            mock_stock = Mock()
            mock_stock.history.return_value = pd.DataFrame({
                'Close': [100, 101, 102, 103, 104]
            })
            mock_stock.info = {
                'marketCap': 1000000000,
                'trailingPE': 15.5,
                'debtToEquity': 0.3,
                'returnOnEquity': 0.12
            }
            mock_ticker.return_value = mock_stock
            
            result = self.engine.fetch_financial_data('AAPL')
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['ticker'], 'AAPL')
            self.assertGreater(result['current_price'], 0)
            self.assertIsInstance(result['volatility'], float)
    
    def test_fetch_financial_data_fallback(self):
        """Test fallback when API fails"""
        with patch('yfinance.Ticker', side_effect=Exception("API Error")):
            result = self.engine.fetch_financial_data('INVALID')
            
            self.assertIsInstance(result, dict)
            self.assertEqual(result['ticker'], 'INVALID')
            self.assertEqual(result['current_price'], 100)  # Default fallback
    
    def test_fetch_macro_data(self):
        """Test macro data generation"""
        result = self.engine.fetch_macro_data()
        
        required_keys = ['gdp_growth', 'inflation_rate', 'unemployment_rate', 
                        'interest_rate', 'vix_index', 'dollar_index']
        
        for key in required_keys:
            self.assertIn(key, result)
            self.assertIsInstance(result[key], (int, float))
    
    def test_fetch_news_sentiment(self):
        """Test news sentiment generation"""
        result = self.engine.fetch_news_sentiment('Apple Inc')
        
        self.assertIn('sentiment_score', result)
        self.assertIn('news_volume', result)
        self.assertIn('recent_events', result)
        self.assertIsInstance(result['recent_events'], list)

class TestCreditScoringEngine(unittest.TestCase):
    """Test credit scoring functionality"""
    
    def setUp(self):
        self.engine = CreditScoringEngine()
        self.sample_financial_data = {
            'debt_to_equity': 0.3,
            'return_on_equity': 0.15,
            'pe_ratio': 12.5,
            'current_ratio': 1.5,
            'profit_margins': 0.12,
            'revenue_growth': 0.08,
            'volatility': 0.25,
            'price_momentum': 0.05,
            'trend_strength': 0.02
        }
        self.sample_macro_data = {
            'gdp_growth': 2.5,
            'inflation_rate': 3.2,
            'unemployment_rate': 4.1,
            'interest_rate': 5.25,
            'vix_index': 18.0,
            'dollar_index': 103.0
        }
        self.sample_sentiment_data = {
            'sentiment_score': 0.15,
            'news_volume': 7
        }
    
    def test_prepare_features(self):
        """Test feature preparation"""
        features = self.engine.prepare_features(
            self.sample_financial_data,
            self.sample_macro_data,
            self.sample_sentiment_data
        )
        
        expected_features = [
            'debt_to_equity_norm', 'roe_score', 'pe_score', 'liquidity_score',
            'profitability_score', 'growth_score', 'volatility_risk',
            'momentum_score', 'trend_score', 'macro_risk', 'interest_risk',
            'market_stress', 'sentiment_boost', 'news_attention'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
    
    def test_calculate_credit_score_range(self):
        """Test credit score is within valid range"""
        features = self.engine.prepare_features(
            self.sample_financial_data,
            self.sample_macro_data,
            self.sample_sentiment_data
        )
        score = self.engine.calculate_credit_score(features)
        
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1000)
        self.assertIsInstance(score, (int, float))
    
    def test_get_feature_explanations(self):
        """Test feature explanations generation"""
        features = self.engine.prepare_features(
            self.sample_financial_data,
            self.sample_macro_data,
            self.sample_sentiment_data
        )
        score = self.engine.calculate_credit_score(features)
        explanations = self.engine.get_feature_explanations(features, score)
        
        self.assertIsInstance(explanations, list)
        for explanation, impact, category in explanations:
            self.assertIsInstance(explanation, str)
            self.assertIsInstance(impact, (int, float))
            self.assertIn(category, ['positive', 'warning'])

class TestNLPEngines(unittest.TestCase):
    """Test NLP processing components"""
    
    def setUp(self):
        self.event_extractor = FinancialEventExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_aggregator = NewsAggregator()
    
    def test_event_extraction(self):
        """Test financial event extraction"""
        test_text = "Apple Inc reports strong quarterly earnings beat expectations with 15% revenue growth"
        events = self.event_extractor.extract_events(test_text)
        
        self.assertIsInstance(events, list)
        if events:
            event = events[0]
            self.assertIn('event_type', event)
            self.assertIn('context', event)
            self.assertIn('sentiment', event)
            self.assertIn('confidence', event)
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality"""
        positive_text = "Strong earnings growth and excellent market performance"
        negative_text = "Significant losses and declining revenue concerns"
        neutral_text = "The company reported quarterly results"
        
        pos_result = self.sentiment_analyzer.analyze_sentiment(positive_text)
        neg_result = self.sentiment_analyzer.analyze_sentiment(negative_text)
        neu_result = self.sentiment_analyzer.analyze_sentiment(neutral_text)
        
        # Check structure
        for result in [pos_result, neg_result, neu_result]:
            self.assertIn('polarity', result)
            self.assertIn('subjectivity', result)
            self.assertIn('classification', result)
            self.assertIn('confidence', result)
        
        # Check sentiment direction
        self.assertGreater(pos_result['polarity'], neu_result['polarity'])
        self.assertLess(neg_result['polarity'], neu_result['polarity'])
    
    def test_risk_signal_detection(self):
        """Test risk signal detection using sentiment analysis"""
        risk_text = "Company faces bankruptcy filing and major debt restructuring"
        safe_text = "Company reports stable operations and strong cash flow"
        
        risk_sentiment = self.sentiment_analyzer.analyze_sentiment(risk_text)
        safe_sentiment = self.sentiment_analyzer.analyze_sentiment(safe_text)
        
        self.assertLess(risk_sentiment['polarity'], safe_sentiment['polarity'])
    
    def test_unstructured_data_processor(self):
        """Test complete unstructured data processing"""
        news_data = self.news_aggregator.fetch_company_news("Test Company", "TEST", 3)
        
        self.assertIsInstance(news_data, list)
        if news_data:
            article = news_data[0]
            self.assertIn('title', article)
            self.assertIn('content', article)
            self.assertIn('published_date', article)

class TestAdvancedModels(unittest.TestCase):
    """Test advanced ML model components"""
    
    def setUp(self):
        self.feature_engineer = AdvancedFeatureEngineer()
        self.model_ensemble = ModelEnsemble()
        
        # Generate sample data
        np.random.seed(42)
        self.sample_prices = np.random.randn(100).cumsum() + 100
        self.sample_financial_data = {
            'grossMargins': 0.4,
            'operatingMargins': 0.15,
            'profitMargins': 0.12,
            'currentRatio': 1.5,
            'debtToEquity': 0.3
        }
    
    def test_technical_indicators(self):
        """Test technical indicator calculation"""
        indicators = self.feature_engineer.create_technical_indicators(self.sample_prices)
        
        expected_indicators = ['sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 
                             'macd', 'rsi', 'bb_upper', 'bb_lower', 'momentum_5']
        
        for indicator in expected_indicators:
            if indicator in indicators:  # Some indicators might not be calculated for short series
                self.assertIsInstance(indicators[indicator], (int, float))
                self.assertFalse(np.isnan(indicators[indicator]))
    
    def test_fundamental_features(self):
        """Test fundamental analysis feature creation"""
        features = self.feature_engineer.create_fundamental_features(self.sample_financial_data)
        
        expected_features = ['gross_margin', 'operating_margin', 'net_margin', 
                           'current_ratio', 'debt_to_equity']
        
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], (int, float))
    
    def test_model_ensemble_initialization(self):
        """Test model ensemble initialization"""
        self.model_ensemble.initialize_models()
        
        expected_models = ['random_forest', 'gradient_boosting', 'xgboost', 
                          'lightgbm', 'neural_network', 'elastic_net']
        
        for model_name in expected_models:
            self.assertIn(model_name, self.model_ensemble.models)
            self.assertIn(model_name, self.model_ensemble.feature_scalers)
    
    def test_model_training_pipeline(self):
        """Test complete model training pipeline"""
        # Generate synthetic training data
        n_samples = 1000
        n_features = 10
        
        X = np.random.randn(n_samples, n_features)
        # Create realistic credit scores (600-900 range)
        y = 700 + np.random.randn(n_samples) * 50
        y = np.clip(y, 600, 900)
        
        self.model_ensemble.initialize_models()
        scores = self.model_ensemble.train_ensemble(X, y, validation_split=0.2)
        
        self.assertIsInstance(scores, dict)
        self.assertTrue(self.model_ensemble.is_trained)
        self.assertGreater(len(self.model_ensemble.model_weights), 0)
        
        # Test prediction
        X_test = np.random.randn(10, n_features)
        predictions, individual_preds = self.model_ensemble.predict(X_test)
        
        self.assertEqual(len(predictions), 10)
        self.assertIsInstance(individual_preds, dict)

class TestIntegrationScenarios(unittest.TestCase):
    """Test complete end-to-end integration scenarios"""
    
    def setUp(self):
        self.data_engine = DataIngestionEngine()
        self.scoring_engine = CreditScoringEngine()
        self.sentiment_analyzer = SentimentAnalyzer()
    
    @patch('yfinance.Ticker')
    def test_complete_scoring_pipeline(self, mock_ticker):
        """Test complete credit scoring pipeline"""
        # Mock financial data
        mock_stock = Mock()
        mock_stock.history.return_value = pd.DataFrame({
            'Close': [100, 101, 102, 103, 104]
        })
        mock_stock.info = {
            'marketCap': 1000000000,
            'trailingPE': 15.0,
            'debtToEquity': 0.3,
            'returnOnEquity': 0.12,
            'profitMargins': 0.1,
            'currentRatio': 1.2
        }
        mock_ticker.return_value = mock_stock
        
        # Run complete pipeline
        financial_data = self.data_engine.fetch_financial_data('AAPL')
        macro_data = self.data_engine.fetch_macro_data()
        sentiment_data = self.data_engine.fetch_news_sentiment('Apple Inc')
        
        features = self.scoring_engine.prepare_features(
            financial_data, macro_data, sentiment_data
        )
        score = self.scoring_engine.calculate_credit_score(features)
        explanations = self.scoring_engine.get_feature_explanations(features, score)
        
        # Validate results
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1000)
        self.assertIsInstance(explanations, list)
    
    def test_data_quality_validation(self):
        """Test data quality and validation"""
        # Test with invalid/missing data
        incomplete_data = {'pe_ratio': None, 'debt_to_equity': -1}
        macro_data = self.data_engine.fetch_macro_data()
        sentiment_data = {'sentiment_score': 0, 'news_volume': 0}
        
        # Should handle missing/invalid data gracefully
        features = self.scoring_engine.prepare_features(
            incomplete_data, macro_data, sentiment_data
        )
        score = self.scoring_engine.calculate_credit_score(features)
        
        # Score should still be valid
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1000)
    
    def test_performance_benchmarks(self):
        """Test performance requirements"""
        import time
        
        # Test scoring latency
        financial_data = self.data_engine._get_default_data('TEST')
        macro_data = self.data_engine.fetch_macro_data()
        sentiment_data = self.data_engine.fetch_news_sentiment('Test Company')
        
        start_time = time.time()
        
        features = self.scoring_engine.prepare_features(
            financial_data, macro_data, sentiment_data
        )
        score = self.scoring_engine.calculate_credit_score(features)
        explanations = self.scoring_engine.get_feature_explanations(features, score)
        
        end_time = time.time()
        latency = end_time - start_time
        
        # Should complete within 1 second
        self.assertLess(latency, 1.0)
    
    def test_error_handling_robustness(self):
        """Test system robustness with various error conditions"""
        # Test with completely invalid ticker
        result = self.data_engine.fetch_financial_data('INVALID_TICKER_123')
        self.assertIsInstance(result, dict)
        self.assertIn('ticker', result)
        
        # Test with extreme values
        extreme_data = {
            'debt_to_equity': 1000,  # Extremely high
            'return_on_equity': -0.5,  # Negative ROE
            'volatility': 5.0,  # Very high volatility
            'pe_ratio': -10  # Invalid P/E
        }
        
        macro_data = self.data_engine.fetch_macro_data()
        sentiment_data = {'sentiment_score': -10, 'news_volume': 1000}
        
        # Should handle extreme values without crashing
        features = self.scoring_engine.prepare_features(
            extreme_data, macro_data, sentiment_data
        )
        score = self.scoring_engine.calculate_credit_score(features)
        
        self.assertIsInstance(score, (int, float))
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1000)

class TestAPIEndpoints(unittest.TestCase):
    """Test API functionality (if implemented)"""
    
    def setUp(self):
        # This would test FastAPI endpoints if implemented
        pass
    
    def test_score_endpoint_response_format(self):
        """Test API response format"""
        # Mock API response format
        expected_response = {
            'ticker': 'AAPL',
            'score': 750,
            'rating': 'A',
            'confidence': 0.85,
            'last_updated': datetime.now().isoformat(),
            'explanations': []
        }
        
        # Validate response structure
        required_fields = ['ticker', 'score', 'rating', 'confidence', 'last_updated']
        for field in required_fields:
            self.assertIn(field, expected_response)

class TestLoadAndStressScenarios(unittest.TestCase):
    """Test system under load and stress conditions"""
    
    def test_concurrent_scoring_requests(self):
        """Test handling multiple concurrent scoring requests"""
        import threading
        import queue
        
        def score_company(ticker, result_queue):
            try:
                engine = CreditScoringEngine()
                data_engine = DataIngestionEngine()
                
                financial_data = data_engine._get_default_data(ticker)
                macro_data = data_engine.fetch_macro_data()
                sentiment_data = data_engine.fetch_news_sentiment(ticker)
                
                features = engine.prepare_features(financial_data, macro_data, sentiment_data)
                score = engine.calculate_credit_score(features)
                
                result_queue.put((ticker, score, 'success'))
            except Exception as e:
                result_queue.put((ticker, None, str(e)))
        
        # Test with multiple concurrent requests
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'JPM']
        result_queue = queue.Queue()
        threads = []
        
        # Start concurrent scoring
        for ticker in tickers:
            thread = threading.Thread(target=score_company, args=(ticker, result_queue))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join(timeout=10)  # 10 second timeout per thread
        
        # Collect results
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())
        
        # Validate all requests completed successfully
        self.assertEqual(len(results), len(tickers))
        for ticker, score, status in results:
            self.assertEqual(status, 'success')
            self.assertIsInstance(score, (int, float))
    
    def test_memory_usage_stability(self):
        """Test memory usage remains stable under load"""
        import gc
        
        # Simple memory test without psutil
        initial_objects = len(gc.get_objects())
        
        engine = CreditScoringEngine()
        data_engine = DataIngestionEngine()
        
        for i in range(100):
            financial_data = data_engine._get_default_data(f'TEST{i}')
            macro_data = data_engine.fetch_macro_data()
            sentiment_data = data_engine.fetch_news_sentiment(f'Test Company {i}')
            
            features = engine.prepare_features(financial_data, macro_data, sentiment_data)
            score = engine.calculate_credit_score(features)
            
            if i % 10 == 0:
                gc.collect()
        
        final_objects = len(gc.get_objects())
        object_increase = final_objects - initial_objects
        
        # Object count increase should be reasonable
        self.assertLess(object_increase, 1000)

# Utility functions for testing
def generate_realistic_financial_data():
    """Generate realistic financial data for testing"""
    return {
        'market_cap': np.random.lognormal(20, 2),  # Realistic market cap distribution
        'pe_ratio': max(1, np.random.gamma(2, 8)),  # Positive P/E ratios
        'debt_to_equity': max(0, np.random.gamma(1.5, 0.2)),  # Reasonable D/E ratios
        'return_on_equity': np.random.normal(0.12, 0.08),  # ROE distribution
        'current_ratio': max(0.1, np.random.gamma(3, 0.5)),  # Liquidity ratios
        'volatility': max(0.05, np.random.gamma(2, 0.1))  # Market volatility
    }

if __name__ == '__main__':
    unittest.main(verbosity=2)