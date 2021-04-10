# Use Python37
FROM python:3.7

# Copy local code to the container image.
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY ./app .

# Install production dependencies.
RUN pip install -r requirements.txt

# Run flask app
CMD ["gunicorn", "-w","1", "-b","0.0.0.0:8080","app:app"]