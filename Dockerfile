# Use a smaller base image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies first to leverage Docker cache
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install dependencies separately for better caching
COPY requirements.txt .

# Install `uv`
RUN pip install uv

RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Run the FastAPI app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
