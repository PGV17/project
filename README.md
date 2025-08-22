# Financial Analysis Application

**Real-time Credit Scoring and Financial Analysis Dashboard**

A sophisticated financial analysis application that provides real-time credit scoring and risk assessment using machine learning, natural language processing, and live financial data integration.

## Features

- **Real-time Credit Scoring**: Advanced algorithms for dynamic credit assessment
- **Multi-Company Analysis**: Support for major US stocks (AAPL, MSFT, GOOGL, TSLA, etc.)
- **Interactive Dashboards**: Professional visualizations with Plotly
- **Risk Factor Analysis**: Comprehensive risk breakdown and explanations
- **Market Sentiment Analysis**: NLP-powered sentiment scoring
- **Explainable AI**: Detailed explanations for credit scoring decisions
- **Auto-refresh Capability**: Real-time data updates every 30 seconds

## Dashboard Components

### 1. Credit Score Gauge

- Real-time credit scores (0-1000 scale)
- Color-coded risk indicators
- Historical trend analysis

### 2. Key Metrics Panel

- Market Capitalization
- P/E Ratio
- Debt-to-Equity Ratio
- Price Volatility
- Market Sentiment Score

### 3. Risk Factor Analysis

- Interactive risk breakdown charts
- Factor contribution analysis
- Investment grade categorization

### 4. Detailed Explanations

- Contributing factor analysis
- Recent news impact
- Improvement recommendations

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly
- **Machine Learning**: Scikit-learn
- **Financial Data**: Yahoo Finance API
- **NLP**: NLTK, TextBlob
- **Web Scraping**: Requests

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Internet connection for real-time data

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/financial-analysis-app.git
   cd financial-analysis-app
   ```

2. **Create virtual environment**

   ```bash
   python -m venv .venv
   ```

3. **Activate virtual environment**

   - Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation**

   ```bash
   python verify_setup.py
   ```

   This will check all dependencies and download required NLTK data automatically.

## Quick Start

1. **Start the application**

   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Select a company** from the dropdown menu

4. **Analyze results** and explore the interactive dashboard

## Supported Companies

The application includes pre-configured analysis for major companies:

- **Apple Inc. (AAPL)**
- **Microsoft Corporation (MSFT)**
- **Alphabet Inc. - Google (GOOGL)**
- **Tesla Inc. (TSLA)**
- **JPMorgan Chase & Co. (JPM)**
- **Bank of America Corp. (BAC)**
- **Exxon Mobil Corporation (XOM)**
- **Johnson & Johnson (JNJ)**
- **Procter & Gamble Co. (PG)**
- **The Coca-Cola Company (KO)**
- **NVIDIA Corporation (NVDA)**
- **Amazon.com Inc. (AMZN)**
- **Meta Platforms Inc. (META)**
- **Berkshire Hathaway Inc. (BRK.B)**
- **Visa Inc. (V)**

## Configuration

### Auto-refresh Settings

- Enable/disable 30-second auto-refresh
- Manual refresh button available
- Real-time data indicators

### Display Options

- Toggle detailed explanations
- Customize visualization themes
- Export analysis results

## Credit Scoring Methodology

The platform uses a sophisticated multi-factor model:

### Financial Factors (60%)

- Debt-to-Equity Ratio
- Return on Equity
- Profit Margins
- Liquidity Ratios
- Revenue Growth

### Market Factors (25%)

- Price Volatility
- Market Momentum
- Trend Analysis
- Market Capitalization

### Sentiment Factors (15%)

- News Sentiment Analysis
- Market Perception
- Social Media Indicators

## Risk Categories

- **AAA (850+)**: Exceptional creditworthiness
- **AA (750-849)**: Very strong financial position
- **A (650-749)**: Strong credit profile
- **BBB (550-649)**: Adequate credit quality
- **Below 550**: Below investment grade

## Technical Architecture

### Core Modules

1. **app.py**: Main Streamlit application
2. **ml_models.py**: Machine learning algorithms
3. **nlp_engine.py**: Natural language processing
4. **tests/**: Unit test suite

### Data Pipeline

1. **Real-time Data Ingestion**: Yahoo Finance API
2. **Data Processing**: Pandas/NumPy transformations
3. **Feature Engineering**: Financial ratio calculations
4. **ML Scoring**: Ensemble model predictions
5. **Visualization**: Interactive Plotly charts

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/
```

Individual module testing:

```bash
python -c "import ml_models; import nlp_engine; print('All modules imported successfully')"
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This application is for educational and research purposes only. It should not be used as the sole basis for investment or credit decisions. Always consult with qualified financial professionals before making financial decisions.

## Data Sources

- **Financial Data**: Yahoo Finance
- **Market Data**: Real-time stock APIs
- **News Sentiment**: Web scraping and NLP analysis

## Support

For support, please open an issue on GitHub or contact the development team.

## Acknowledgments

- Yahoo Finance for real-time financial data
- Streamlit for the amazing web framework
- Plotly for interactive visualizations
- The open-source community for excellent libraries

---

**Built with care for the financial analysis community**
