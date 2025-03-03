# Use an official Python image as a base
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy all files to the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the FastAPI port
EXPOSE 8005

# Run FastAPI with Uvicorn
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8005"]
