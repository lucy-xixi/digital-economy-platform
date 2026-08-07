# Digital Economy Intelligence Platform (DEIP)
FROM python:3.11-slim

LABEL maintainer="DEIP Team"
LABEL description="Digital Economy Intelligence Platform - Enterprise Demo"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Ensure data / database / log directories exist
RUN mkdir -p data/raw data/processed database logs

# Initialize sample data and compute indicators on first build (optional)
ENV PYTHONPATH=/app
ENV DEIP_ENV=production

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# Bootstrap pipeline then launch Streamlit dashboard
CMD ["sh", "-c", "python -m src.bootstrap && streamlit run dashboard/app.py --server.port=8501 --server.address=0.0.0.0"]
