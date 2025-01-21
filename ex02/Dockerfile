# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PyTorch, OpenCV, FAISS, and FFmpeg
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install FFmpeg (version 4.4.2-0ubuntu0.22.04.1)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg=7:4.4.2-0ubuntu0.22.04.1 \
    && rm -rf /var/lib/apt/lists/*

# Install yt-dlp (version 2024.10.22)
RUN pip install yt-dlp==2024.10.22

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . .

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

# Run the Joint trajectories application
RUN python3 getPoseLandmarks.py -i input/DftBUdHgr9Q.mp4 -o output/trajectories.csv

# Run the cycle detection application
RUN python3 CycleDetection.py -i output/trajectories.csv -o output/cycles.json
