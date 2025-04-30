FROM python:3.11-slim

# Install system deps for PyMuPDF
RUN apt-get update && apt-get install -y build-essential libmupdf-dev && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose port 8000
ENV PORT=8000
<<<<<<< HEAD
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
=======
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> e96e130 (Add main.py and requirements.txt, update Dockerfile)
