FROM python:3.11-slim

WORKDIR /app

# System dependencies for PDF processing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create runtime directories
RUN mkdir -p data/uploads data/vectors data/reports

# HF Spaces expects port 7860
EXPOSE 7860

# Run FastAPI via uvicorn (serves HTML frontend + API)
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
