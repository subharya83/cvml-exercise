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

# Setup test 
RUN yt-dlp -S ext:mp4:m4a https://www.youtube.com/watch?v=rTDaZoDDW5g -o input/rTDaZoDDW5g.mp4
RUN ffmpeg -hide_banner -thread_queue_size 8192 -i input/rTDaZoDDW5g.mp4 input/frames/%06d.jpg
RUN ffmpeg -i input/rTDaZoDDW5g.mp4 -ss 00:00:20 -to 00:00:30 input/test.mp4
RUN ffmpeg -i input/test.mp4 input/testframes/%06d.jpg 

# Run tests
RUN python3 getSimilarity.py -q ./input/testfranes/000010.jpg -t ./input/testframes/000011.jpg
RUN python3 buildIndex.py -i ./input/testframes/ -o ./output/testIndexF.json
RUN python3 queryDataset.py -q ./input/testframes/000100.jpg -i ./output/testIndexF.json
