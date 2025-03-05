# Use an NVIDIA CUDA base image with Python
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV WEIGHTS_DIR="/weights"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3-pip \
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
    srt \
    torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Set the entrypoint to run the script
ENTRYPOINT ["python", "Xscribe.py"]