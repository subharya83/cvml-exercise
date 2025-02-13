# Use an official Python runtime as the base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WEIGHTS_DIR="/weights"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a directory for the app and weights
RUN mkdir -p /app /weights

# Set the working directory
WORKDIR /app

# Copy the script into the container
COPY Xscribe.py /app/Xscribe.py

# Install Python dependencies
RUN pip install --no-cache-dir \
    openai-whisper \
    pyannote.audio \
    srt

# Set the entrypoint to run the script
ENTRYPOINT ["python", "Xscribe.py"]