# Use a smaller base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -fsSL https://astral.sh/uv/install.sh | sh

# Verify uv installation
RUN uv --version && ls -lah /root/.local/bin/

# Copy dependencies first
COPY requirements.txt .

# Verify requirements.txt exists
RUN ls -lah /app/requirements.txt

# Install dependencies using uv
RUN uv pip install -r requirements.txt --verbose

# Copy the rest of the application
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI using uvicorn
CMD ["uv", "pip", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
