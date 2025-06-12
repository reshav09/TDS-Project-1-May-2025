# Use slim Python 3.11 image
FROM python:3.11-slim

# Install OS-level dependencies (for OCR)
RUN apt-get update && apt-get install -y tesseract-ocr && rm -rf /var/lib/apt/lists/*

# Set working directory to your app folder
WORKDIR /app/app

# Copy all files from your local app/ folder into /app/app inside container
COPY app/ .

# Install Python dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Set environment variables (if needed)
ENV PYTHONUNBUFFERED=1

# Run your app directly with Python
CMD ["python", "index.py"]
