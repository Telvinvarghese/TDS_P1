# Use a smaller base image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PATH="/root/.local/bin/:$PATH"

# Set the working directory
WORKDIR /

# Install system dependencies (including Node.js and npm)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates nodejs npm \
    wget sqlite3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Verify system installations
RUN node -v && npm -v && npx -v

RUN npm install -g prettier@3.4.2

# Copy dependencies file first (Docker caching optimization)
COPY requirements.txt .

# Ensure requirements.txt exists
# RUN test -f requirements.txt

# Install uv
RUN curl -fsSL https://astral.sh/uv/install.sh | sh

# Verify uv installation
RUN uv --version && ls -lah /root/.local/bin/

# Install Python dependencies
RUN uv pip install --system -r requirements.txt --no-cache-dir

# Set AIPROXY_TOKEN
ARG AIPROXY_TOKEN
ENV AIPROXY_TOKEN=${AIPROXY_TOKEN}

RUN mkdir -p /data

# Copy the rest of the application
COPY main.py prompts.py .

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI using uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
