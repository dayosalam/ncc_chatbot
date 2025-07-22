# Use an official Python runtime as a parent image
FROM python:3.10.2-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application folder
COPY ./my-app/ .

# The pirate-speak package is a local dependency that needs to be installed.
RUN pip install ./packages/pirate-speak

# Change the working directory to the app folder where server.py is located
WORKDIR /app/app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]