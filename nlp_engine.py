"""
NLP Engine for Financial Data Processing
"""

import pandas as pd
import numpy as np
import re
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except:
    pass

class FinancialEventExtractor:
    """Extract and classify financial events from text"""
    
    def __init__(self):
        self.event_patterns = {
            'earnings': [
                r'earnings?\s+report',
                r'quarterly\s+results',
                r'q[1-4]\s+earnings',
                r'earnings?\s+call',
                r'financial\s+results'
            ],
            'merger_acquisition': [
                r'merger?\s+with',
                r'acquire[ds]?\s+by',
                r'acquisition\s+of',
                r'takeover\s+bid',
                r'buyout\s+deal'
            ],
            'debt_financing': [
                r'debt\s+restructuring',
                r'bond\s+issuance',
                r'credit\s+facility',
                r'loan\s+agreement',
                r'refinancing\s+deal',
                r'bankruptcy\s+filing',
                r'chapter\s+11'
            ],
            'leadership_change': [
                r'ceo\s+resign',
                r'chief\s+executive\s+steps\s+down',
                r'new\s+ceo',
                r'management\s+change',
                r'board\s+of\s+directors'
            ],
            'regulatory': [
                r'regulatory\s+approval',
                r'sec\s+investigation',
                r'compliance\s+violation',
                r'regulatory\s+fine',
                r'government\s+probe'
            ],
            'product_launch': [
                r'product\s+launch',
                r'new\s+product',
                r'patent\s+approval',
                r'fda\s+approval',
                r'clinical\s+trial'
            ],
            'guidance_revision': [
                r'guidance\s+raised',
                r'guidance\s+lowered',
                r'outlook\s+revised',
                r'forecast\s+updated',
                r'projections\s+changed'
            ]
        }
        
        self.sentiment_modifiers = {
            'positive': ['strong', 'excellent', 'outstanding', 'beat', 'exceed', 'growth', 'improvement'],
            'negative': ['weak', 'poor', 'decline', 'loss', 'miss', 'below', 'concern', 'risk']
        }
        
    def extract_events(self, text):
        """Extract financial events from text"""
        text_lower = text.lower()
        detected_events = []
        
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower)
                for match in matches:
                    start = max(0, match.start() - 50)
                    end = min(len(text), match.end() + 50)
                    context = text[start:end]
                    
                    sentiment = self._analyze_context_sentiment(context)
                    
                    detected_events.append({
                        'event_type': event_type,
                        'pattern': pattern,
                        'context': context.strip(),
                        'sentiment': sentiment,
                        'position': match.start(),
                        'confidence': self._calculate_confidence(pattern, context)
                    })
        
        return detected_events
    
    def _analyze_context_sentiment(self, context):
        """Analyze sentiment in the context of an event"""
        context_lower = context.lower()
        positive_score = sum(1 for word in self.sentiment_modifiers['positive'] if word in context_lower)
        negative_score = sum(1 for word in self.sentiment_modifiers['negative'] if word in context_lower)
        
        if positive_score > negative_score:
            return 'positive'
        elif negative_score > positive_score:
            return 'negative'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, pattern, context):
        """Calculate confidence score for event detection"""
        # Simple heuristic based on pattern specificity and context length
        pattern_specificity = len(pattern.split()) / 10.0
        context_relevance = min(len(context.split()) / 20.0, 1.0)
        return min(pattern_specificity + context_relevance, 1.0)

class SentimentAnalyzer:
    """Sentiment analysis for financial text"""
    
    def __init__(self):
        self.financial_lexicon = self._load_financial_lexicon()
        self.stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
    def _load_financial_lexicon(self):
        """Load financial-specific sentiment lexicon"""
        return {
            'growth': 2, 'profit': 2, 'revenue': 1, 'earnings': 1, 'strong': 2,
            'outperform': 2, 'beat': 2, 'exceed': 2, 'upgrade': 2, 'bullish': 2,
            'expansion': 1, 'acquisition': 1, 'partnership': 1, 'innovation': 1,
            
            'loss': -2, 'decline': -2, 'fall': -1, 'drop': -1, 'weak': -2,
            'underperform': -2, 'miss': -2, 'downgrade': -2, 'bearish': -2,
            'bankruptcy': -3, 'default': -3, 'investigation': -2, 'lawsuit': -2,
            'recession': -2, 'crisis': -3, 'volatility': -1, 'uncertainty': -1
        }
    
    def analyze_sentiment(self, text):
        """Comprehensive sentiment analysis"""
        blob = TextBlob(text)
        textblob_sentiment = blob.sentiment
        
        financial_sentiment = self._calculate_financial_sentiment(text)
        
        combined_polarity = (textblob_sentiment.polarity + financial_sentiment) / 2
        
        # Confidence based on text length and subjectivity
        confidence = min(len(text.split()) / 50.0, 1.0) * (1 - textblob_sentiment.subjectivity)
        
        return {
            'polarity': combined_polarity,
            'subjectivity': textblob_sentiment.subjectivity,
            'financial_score': financial_sentiment,
            'confidence': confidence,
            'classification': self._classify_sentiment(combined_polarity)
        }
    
    def _calculate_financial_sentiment(self, text):
        """Calculate sentiment using financial lexicon"""
        tokens = word_tokenize(text.lower())
        sentiment_score = 0
        word_count = 0
        
        for token in tokens:
            if token not in self.stopwords:
                lemmatized = self.lemmatizer.lemmatize(token)
                if lemmatized in self.financial_lexicon:
                    sentiment_score += self.financial_lexicon[lemmatized]
                    word_count += 1
        
        return sentiment_score / max(word_count, 1)
    
    def _classify_sentiment(self, polarity):
        """Classify sentiment into categories"""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'

class NewsAggregator:
    """Aggregate and process news from multiple sources"""
    
    def __init__(self, api_keys=None):
        self.api_keys = api_keys or {}
        self.base_urls = {
            'newsapi': 'https://newsapi.org/v2/',
            'alpha_vantage': 'https://www.alphavantage.co/query'
        }
        
    def fetch_company_news(self, company_name, ticker, days_back=7):
        """Fetch recent news for a company"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        simulated_news = self._generate_simulated_news(company_name, ticker, days_back)
        
        return simulated_news
    
    def _generate_simulated_news(self, company_name, ticker, days_back):
        """Generate realistic simulated news data"""
        news_templates = [
            f"{company_name} reports quarterly earnings beat with strong revenue growth",
            f"Analysts raise price target for {ticker} following positive guidance",
            f"{company_name} announces strategic partnership to expand market reach",
            f"Institutional investors increase holdings in {ticker}",
            f"{company_name} faces regulatory scrutiny over compliance practices",
            f"Market volatility affects {company_name} stock performance",
            f"{company_name} CEO optimistic about future growth prospects",
            f"Supply chain disruptions impact {company_name} operations"
        ]
        
        news_articles = []
        
        for i in range(min(10, days_back * 2)):
            article_date = datetime.now() - timedelta(days=np.random.randint(0, days_back))
            
            template = np.random.choice(news_templates)
            sentiment_bias = np.random.choice(['positive', 'neutral', 'negative'], p=[0.4, 0.3, 0.3])
            
            if sentiment_bias == 'positive':
                template = template.replace('faces regulatory scrutiny', 'receives regulatory approval')
                template = template.replace('Market volatility affects', 'Strong market performance boosts')
            elif sentiment_bias == 'negative':
                template = template.replace('reports quarterly earnings beat', 'disappoints with quarterly earnings miss')
                template = template.replace('raise price target', 'lower price target')
            
            news_articles.append({
                'title': template,
                'published_date': article_date,
                'source': np.random.choice(['Reuters', 'Bloomberg', 'CNBC', 'MarketWatch', 'Yahoo Finance']),
                'url': f"https://example.com/news/{i}",
                'content': f"Full article content for: {template}..."
            })
        
        return sorted(news_articles, key=lambda x: x['published_date'], reverse=True)

class RiskSignalDetector:
    """Detect risk signals from unstructured data"""
    
    def __init__(self):
        self.risk_keywords = {
            'credit_risk': [
                'default', 'bankruptcy', 'chapter 11', 'debt restructuring',
                'covenant breach', 'payment delay', 'credit downgrade'
            ],
            'operational_risk': [
                'cybersecurity breach', 'data breach', 'system failure',
                'supply chain disruption', 'regulatory violation'
            ],
            'market_risk': [
                'market crash', 'volatility spike', 'liquidity crisis',
                'currency devaluation', 'interest rate shock'
            ],
            'reputation_risk': [
                'scandal', 'investigation', 'lawsuit', 'fraud',
                'whistleblower', 'ethical violation'
            ],
            'strategic_risk': [
                'market share loss', 'competitive threat', 'technology disruption',
                'regulatory change', 'management turnover'
            ]
        }
        
        self.severity_indicators = {
            'high': ['severe', 'major', 'significant', 'massive', 'unprecedented'],
            'medium': ['moderate', 'notable', 'considerable', 'substantial'],
            'low': ['minor', 'slight', 'small', 'limited', 'manageable']
        }
    
    def detect_risk_signals(self, text_data):
        """Detect and classify risk signals"""
        risk_signals = []
        
        for text in text_data if isinstance(text_data, list) else [text_data]:
            text_lower = text.lower()
            
            for risk_category, keywords in self.risk_keywords.items():
                for keyword in keywords:
                    if keyword in text_lower:
                        severity = self._assess_severity(text_lower, keyword)
                        context = self._extract_context(text, keyword)
                        
                        risk_signals.append({
                            'risk_category': risk_category,
                            'keyword': keyword,
                            'severity': severity,
                            'context': context,
                            'detected_at': datetime.now(),
                            'confidence': self._calculate_risk_confidence(text_lower, keyword)
                        })
        
        return risk_signals
    
    def _assess_severity(self, text, keyword):
        """Assess the severity of a detected risk"""
        for severity, indicators in self.severity_indicators.items():
            if any(indicator in text for indicator in indicators):
                return severity
        return 'medium'  # Default severity
    
    def _extract_context(self, text, keyword, window=50):
        """Extract context around the risk keyword"""
        keyword_pos = text.lower().find(keyword.lower())
        if keyword_pos == -1:
            return text[:100]
        
        start = max(0, keyword_pos - window)
        end = min(len(text), keyword_pos + len(keyword) + window)
        return text[start:end].strip()
    
    def _calculate_risk_confidence(self, text, keyword):
        """Calculate confidence in risk detection"""
        # Simple heuristic based on keyword specificity and context
        keyword_specificity = len(keyword.split()) * 0.2
        context_support = min(text.count(keyword) * 0.1, 0.5)
        return min(keyword_specificity + context_support, 1.0)

class UnstructuredDataProcessor:
    """Main processor for all unstructured data"""
    
    def __init__(self):
        self.event_extractor = FinancialEventExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_aggregator = NewsAggregator()
        self.risk_detector = RiskSignalDetector()
        
    def process_company_data(self, company_name, ticker, days_back=7):
        """Process all unstructured data for a company"""
        # Fetch news data
        news_articles = self.news_aggregator.fetch_company_news(company_name, ticker, days_back)
        
        results = {
            'company': company_name,
            'ticker': ticker,
            'processed_at': datetime.now(),
            'articles_processed': len(news_articles),
            'overall_sentiment': {'polarity': 0, 'confidence': 0},
            'events': [],
            'risk_signals': [],
            'summary_metrics': {}
        }
        
        all_text = []
        sentiment_scores = []
        
        for article in news_articles:
            # Combine title and content
            full_text = f"{article['title']} {article.get('content', '')}"
            all_text.append(full_text)
            
            # Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_sentiment(full_text)
            sentiment_scores.append(sentiment['polarity'])
            
            # Extract events
            events = self.event_extractor.extract_events(full_text)
            for event in events:
                event['source'] = article['source']
                event['published_date'] = article['published_date']
                results['events'].append(event)
            
            # Detect risk signals
            risks = self.risk_detector.detect_risk_signals(full_text)
            results['risk_signals'].extend(risks)
        
        # Calculate overall sentiment
        if sentiment_scores:
            results['overall_sentiment'] = {
                'polarity': np.mean(sentiment_scores),
                'volatility': np.std(sentiment_scores),
                'confidence': min(len(sentiment_scores) / 10.0, 1.0)
            }
        
        # Generate summary metrics
        results['summary_metrics'] = self._generate_summary_metrics(results)
        
        return results
    
    def _generate_summary_metrics(self, results):
        """Generate summary metrics from processed data"""
        metrics = {
            'sentiment_score': results['overall_sentiment']['polarity'],
            'sentiment_volatility': results['overall_sentiment'].get('volatility', 0),
            'event_count': len(results['events']),
            'risk_signal_count': len(results['risk_signals']),
            'news_attention_score': min(results['articles_processed'] / 5.0, 1.0)
        }
        
        # Event type distribution
        event_types = {}
        for event in results['events']:
            event_type = event['event_type']
            event_types[event_type] = event_types.get(event_type, 0) + 1
        metrics['event_distribution'] = event_types
        
        # Risk category distribution
        risk_categories = {}
        for risk in results['risk_signals']:
            risk_cat = risk['risk_category']
            risk_categories[risk_cat] = risk_categories.get(risk_cat, 0) + 1
        metrics['risk_distribution'] = risk_categories
        
        # Calculate composite risk score
        risk_weights = {
            'credit_risk': 0.3,
            'operational_risk': 0.2,
            'market_risk': 0.2,
            'reputation_risk': 0.15,
            'strategic_risk': 0.15
        }
        
        composite_risk = 0
        for risk_cat, count in risk_categories.items():
            weight = risk_weights.get(risk_cat, 0.1)
            composite_risk += count * weight
        
        metrics['composite_risk_score'] = min(composite_risk, 1.0)
        
        return metrics

class RealTimeTextStreamProcessor:
    """Process real-time text streams for immediate risk detection"""
    
    def __init__(self, alert_thresholds=None):
        self.processor = UnstructuredDataProcessor()
        self.alert_thresholds = alert_thresholds or {
            'high_risk': 0.7,
            'negative_sentiment': -0.5,
            'event_spike': 5
        }
        self.alert_callbacks = []
        
    def add_alert_callback(self, callback_func):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback_func)
    
    def process_stream(self, text_stream, company_info):
        """Process incoming text stream"""
        results = []
        
        for text_chunk in text_stream:
            # Quick sentiment analysis
            sentiment = self.processor.sentiment_analyzer.analyze_sentiment(text_chunk)
            
            # Event detection
            events = self.processor.event_extractor.extract_events(text_chunk)
            
            # Risk detection
            risks = self.processor.risk_detector.detect_risk_signals(text_chunk)
            
            chunk_result = {
                'timestamp': datetime.now(),
                'company': company_info.get('name', 'Unknown'),
                'ticker': company_info.get('ticker', ''),
                'sentiment': sentiment,
                'events': events,
                'risks': risks,
                'text_sample': text_chunk[:200] + '...' if len(text_chunk) > 200 else text_chunk
            }
            
            # Check for alerts
            self._check_alerts(chunk_result)
            
            results.append(chunk_result)
        
        return results
    
    def _check_alerts(self, chunk_result):
        """Check if alerts should be triggered"""
        alerts = []
        
        # High risk alert
        if len(chunk_result['risks']) > 0:
            max_confidence = max([r['confidence'] for r in chunk_result['risks']])
            if max_confidence > self.alert_thresholds['high_risk']:
                alerts.append({
                    'type': 'high_risk',
                    'message': f"High-confidence risk signal detected for {chunk_result['company']}",
                    'data': chunk_result['risks']
                })
        
        # Negative sentiment alert
        if chunk_result['sentiment']['polarity'] < self.alert_thresholds['negative_sentiment']:
            alerts.append({
                'type': 'negative_sentiment',
                'message': f"Strong negative sentiment detected for {chunk_result['company']}",
                'data': chunk_result['sentiment']
            })
        
        # Event spike alert
        if len(chunk_result['events']) > self.alert_thresholds['event_spike']:
            alerts.append({
                'type': 'event_spike',
                'message': f"High event activity detected for {chunk_result['company']}",
                'data': chunk_result['events']
            })
        
        # Trigger callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert, chunk_result)
                except Exception as e:
                    print(f"Alert callback error: {e}")

# Example usage and testing functions
def demo_nlp_pipeline():
    """Demonstrate the NLP pipeline capabilities"""
    processor = UnstructuredDataProcessor()
    
    # Sample news text
    sample_news = [
        "Apple Inc. reported strong quarterly earnings beating analyst expectations by 15%. The company showed robust iPhone sales growth and expanding services revenue, leading to positive market reaction.",
        "Tesla faces regulatory investigation over autopilot safety concerns. The National Highway Traffic Safety Administration has launched a probe following multiple accident reports involving the autonomous driving feature.",
        "Microsoft announces strategic partnership with OpenAI, investing additional $10 billion in artificial intelligence development. The collaboration aims to accelerate AI innovation across enterprise applications."
    ]
    
    print("=== NLP Pipeline Demo ===")
    
    for i, text in enumerate(sample_news, 1):
        print(f"\n--- Article {i} ---")
        print(f"Text: {text}")
        
        # Sentiment analysis
        sentiment = processor.sentiment_analyzer.analyze_sentiment(text)
        print(f"Sentiment: {sentiment['classification']} (polarity: {sentiment['polarity']:.3f})")
        
        # Event extraction
        events = processor.event_extractor.extract_events(text)
        if events:
            print("Events detected:")
            for event in events:
                print(f"  - {event['event_type']}: {event['context'][:50]}...")
        
        # Risk detection
        risks = processor.risk_detector.detect_risk_signals(text)
        if risks:
            print("Risk signals:")
            for risk in risks:
                print(f"  - {risk['risk_category']}: {risk['keyword']} (severity: {risk['severity']})")

if __name__ == "__main__":
    demo_nlp_pipeline()