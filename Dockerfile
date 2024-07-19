# Use an official Python runtime as a parent image
FROM python:3.10-slim-bullseye

# Set the working directory in the container
WORKDIR /app
RUN chmod 777 /app

ENV PYTHONPATH="/app"

# Copy only the requirements.txt first to leverage Docker cache
COPY requirements.txt .

RUN apt-get update && apt-get install -y \
    libgomp1 ffmpeg\
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the codebase into the image
COPY . .

# Command to run the application
CMD ["chainlit", "run", "./main.py","-w","--port","15433"]