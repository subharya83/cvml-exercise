FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for some Python packages
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and source code
COPY requirements.txt .
COPY convertQA.py .
COPY input .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Create directory for BERT model weights
RUN mkdir -p ./weights
RUN mkdir -p ./output

# Run tests
RUN python3 convertQA.py -i input/3-Kings.pdf -o output/3-Kings.json
RUN python3 convertQA.py -i input/Argo.htm -o output/Argo.csv