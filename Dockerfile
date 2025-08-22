# Dockerfile for Financial Analysis App
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Copy application code
COPY . .

# Create streamlit config directory
RUN mkdir -p ~/.streamlit

# Create streamlit config
RUN echo "\
[server]\n\
headless = true\n\
enableCORS = false\n\
port = 8501\n\
[browser]\n\
gatherUsageStats = false\n\
" > ~/.streamlit/config.toml

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
