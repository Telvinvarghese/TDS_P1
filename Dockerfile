# Use a smaller base image
FROM python:3.9-slim

# Set environment variables for better logging and performance
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install uv, then remove the installer
RUN curl -fsSL https://astral.sh/uv/install.sh -o uv-installer.sh \
    && sh uv-installer.sh \
    && rm uv-installer.sh

# Copy dependency files first for better caching
COPY requirements.txt .

# Upgrade pip and install dependencies efficiently
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Start the FastAPI app using uv
CMD ["uv", "serve", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

