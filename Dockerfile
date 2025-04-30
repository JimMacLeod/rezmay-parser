# ────────────────────────────
# 1. Use slim python image
# ────────────────────────────
FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
# Force the stdout/stderr streams to line-buffered (useful for logs)
ENV PYTHONUNBUFFERED=1

# ────────────────────────────
# 2. Install minimal build tools (avoid OOM on Railway)
# ────────────────────────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

# ────────────────────────────
# 3. Set work dir & copy files
# ────────────────────────────
WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# ────────────────────────────
# 4. Expose & run
# ────────────────────────────
ENV PORT=8000
EXPOSE ${PORT}

# If your entry-point file is main.py and the FastAPI/Starlette app is usually named “app”
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]