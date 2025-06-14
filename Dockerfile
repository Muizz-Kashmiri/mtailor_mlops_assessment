# Base image with Python 3.11.12
FROM python:3.11.12-slim-bullseye

# Install dependencies for system libraries (if required for your environment)
RUN apt-get update && apt-get install -y \
    dumb-init \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-ca-certificates
# Set environment variables
ENV API_KEY=MTAILOR
# Copy your application code into the container
COPY . /app
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port for Flask app (default port 5000 for Flask)
EXPOSE 8000

# Use dumb-init as init system (to handle signals correctly)
CMD ["dumb-init", "--", "python", "app.py"]
