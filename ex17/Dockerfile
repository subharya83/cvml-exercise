# Use an official PyTorch runtime as a parent image
FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Run Train.py when the container launches
CMD ["python", "Train.py"]