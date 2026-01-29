# Step 1: Use a lightweight Python image
FROM python:3.11-slim

# Step 2: Set the working directory
WORKDIR /app

# Step 3: Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Step 4: Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 5: Copy the application code
# This copies everything from your local 'app' folder into the container's '/app'
COPY app/ .

# Step 6: Expose the port
EXPOSE 8000

# Step 7: Run the FastAPI app
# Note: 'main:app' assumes main.py is directly inside the app/ folder
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]