FROM python:3.11-slim

WORKDIR /app

# System dependencies (minimal — PaddleOCR excluded on HF free tier)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create runtime directories
RUN mkdir -p data/uploads data/vectors data/reports

# Expose Streamlit port (HF Spaces expects 7860)
EXPOSE 7860

# Run Streamlit on port 7860
CMD ["python", "-m", "streamlit", "run", "app/frontend/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
