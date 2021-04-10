# Use Python37
FROM python:3.7

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install production dependencies.
RUN pip install -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Run flask app
CMD ["python", "app.py"]