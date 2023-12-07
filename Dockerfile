# Use the official Python 3.11 slim base image
FROM python:3.11-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies specified in requirements.txt without caching to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Install the Gunicorn web server to serve the Flask application
RUN pip install gunicorn

# Expose port 5000 to allow external access to the Flask application
EXPOSE 5000

# Specify the entry point for the container using Gunicorn to run the Flask application
ENTRYPOINT ["gunicorn", "flk.py", "-b", "0.0.0.0:5000"]


