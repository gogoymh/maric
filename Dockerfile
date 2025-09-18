# FROM nvcr.io/nvidia/pytorch:22.12-py3
FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /app

# Copy requirements file first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .