# Use the official Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements and application code into the container
COPY requirements.txt ./
# Copy specific files into the container
COPY model.py ./
COPY 1model_best.pth.tar ./
COPY 2model_best.pth.tar ./

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Flask will run on
EXPOSE 5000

# Define the environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000

# Run the Flask application
CMD ["python", "app.py"]
