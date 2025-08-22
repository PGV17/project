# Deployment Guide

## Quick Deployment Options

### Option 1: Streamlit Cloud (Easiest - FREE)

**Best for: Public sharing, demos, portfolios**

1. **Upload to GitHub** (if not already done):

   ```bash
   git init
   git add .
   git commit -m "Financial analysis app"
   git push origin main
   ```

2. **Deploy to Streamlit Cloud**:

   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `financial-analysis-app`
   - Main file path: `app.py`
   - Click "Deploy!"

3. **Access your app**:
   - Your app will be live at: `https://yourname-financial-analysis-app-app-xxxxx.streamlit.app`
   - Share this URL with anyone!

### Option 2: Heroku (Professional Hosting)

**Best for: Production apps, custom domains**

1. **Create Heroku files**:

   ```bash
   # We'll create these files for you below
   ```

2. **Deploy**:
   ```bash
   heroku create your-financial-app
   git push heroku main
   ```

### Option 3: Railway (Modern & Simple)

**Best for: Fast deployment, great developer experience**

1. **Connect GitHub to Railway**:
   - Go to [railway.app](https://railway.app)
   - Connect your GitHub repository
   - Railway auto-detects Python and deploys!

### Option 4: Docker (Any Platform)

**Best for: Consistent deployment across environments**

```bash
# Build the image
docker build -t financial-analysis-app .

# Run the container
docker run -p 8501:8501 financial-analysis-app

# Or use docker-compose
docker-compose up
```

## Recommended: Streamlit Cloud (Free)

**Step-by-step guide for easiest deployment:**

1. **Upload your code to GitHub**:

   ```bash
   git init
   git add .
   git commit -m "Financial analysis dashboard"
   git branch -M main
   git remote add origin https://github.com/yourusername/financial-analysis-app.git
   git push -u origin main
   ```

2. **Deploy to Streamlit Cloud**:

   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Click "Sign in" and connect your GitHub account
   - Click "New app"
   - Repository: Select your `financial-analysis-app` repo
   - Branch: `main`
   - Main file path: `app.py`
   - Click "Deploy!"

3. **Your app will be live in 2-3 minutes at**:
   `https://yourusername-financial-analysis-app-app-xxxxx.streamlit.app`

4. **Share your app**:
   - Copy the URL and share with anyone
   - No server management required
   - Automatic updates when you push to GitHub

## Production Deployment (Heroku)

**For professional hosting with custom domains:**

1. **Install Heroku CLI** and login:

   ```bash
   heroku login
   ```

2. **Create Heroku app**:

   ```bash
   heroku create your-financial-app-name
   ```

3. **Deploy**:

   ```bash
   git push heroku main
   ```

4. **Open your app**:
   ```bash
   heroku open
   ```

## Quick Deploy (Railway)

**Modern deployment platform:**

1. Go to [railway.app](https://railway.app)
2. Connect your GitHub repository
3. Railway automatically detects and deploys your Python app
4. Get a live URL in seconds!

## Environment Variables

For production deployments, you can set these environment variables:

```bash
STREAMLIT_SERVER_HEADLESS=true
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
STREAMLIT_SERVER_PORT=8501
```

## Post-Deployment Checklist

- [ ] App loads successfully
- [ ] All company data loads properly
- [ ] Charts and visualizations work
- [ ] Credit scoring calculations function
- [ ] No console errors
- [ ] Mobile responsiveness verified

## Access Your Deployed App

Once deployed, your financial analysis dashboard will be accessible to anyone with the URL. Users can:

- Analyze real-time financial data for major companies
- View interactive credit scoring dashboards
- Explore risk factor breakdowns
- Access professional financial visualizations

## Tips

- **Streamlit Cloud**: Best for demos and sharing
- **Heroku**: Best for production with custom features
- **Docker**: Best for enterprise/on-premise deployment
- **Railway**: Best for modern, fast deployment

Choose the option that best fits your needs!

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 4. Heroku Deployment

Create `Procfile`:

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

## Configuration

### Environment Variables

- `PORT`: Server port (default: 8501)
- `STREAMLIT_SERVER_HEADLESS`: Set to true for production

### Streamlit Configuration

Edit `.streamlit/config.toml` for custom settings:

- Theme colors
- Server configuration
- Browser settings

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change port in config.toml
2. **Missing dependencies**: Run `pip install -r requirements.txt`
3. **NLTK data missing**: Run NLTK download commands
4. **Memory issues**: Increase server limits in config

### Performance Optimization

- Enable caching with `@st.cache_data`
- Use session state efficiently
- Optimize data fetching frequency
- Consider data compression

## Monitoring

### Health Checks

- Application responsiveness
- Data source connectivity
- Memory usage
- Error rates

### Logging

- Streamlit built-in logging
- Custom application logs
- Error tracking and reporting
