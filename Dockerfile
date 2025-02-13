# Use a smaller base image
FROM python:3.9-slim

# Set environment variables for better logging and performance
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies efficiently
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Download and install uv, then remove the installer
RUN curl -fsSL https://astral.sh/uv/install.sh -o uv-installer.sh \
    && sh uv-installer.sh \
    && rm uv-installer.sh

# Copy only dependencies file first to leverage Docker cache
COPY requirements.txt .

# Use uv to install dependencies
RUN uv pip install -r requirements.txt

# Now copy the rest of the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Start FastAPI using uv
CMD ["uv", "pip", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

