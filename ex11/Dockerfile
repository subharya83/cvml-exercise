# Use an official PyTorch image as the base image
FROM pytorch/pytorch:latest-cuda12.2-cudnn8-runtime

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Set the working directory
WORKDIR /workspace

# Copy your code and data into the container
COPY . .

# Default command (can be overridden when running the container)
CMD ["bash"]