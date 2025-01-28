# Use an official Ubuntu base image
FROM ubuntu:20.04

# Set environment variables to avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libeigen3-dev \
    python3 \
    python3-pip \
    wget \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install Boost for C++ serialization
RUN apt-get update && apt-get install -y libboost-all-dev

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . .

# Build any C++ components if necessary
RUN mkdir build && cd build && \
    cmake .. && \
    make

# Set the default command to run the Python training script
CMD ["python3", "Train.py"]