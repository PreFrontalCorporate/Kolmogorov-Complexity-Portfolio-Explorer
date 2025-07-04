# Use the official Python image from Docker Hub
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 to the outside world
EXPOSE 5000

# Run the Flask app
CMD ["python", "api/optimize.py"]
