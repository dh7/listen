# Use an official Python runtime with CUDA as a parent image
FROM python:3.8-alpine

# Copy requirements.txt to the docker image and install Python dependencies
COPY ./requirements.txt /listen/requirements.txt
WORKDIR /listen
RUN pip install --no-cache-dir --upgrade -r /listen/requirements.txt
COPY ./app /listen/app

# Define environment variable
#ENV NAME SpeakerDiarization

# Make port 80 available to the world outside this container
EXPOSE 80

# Run app.py with Uvicorn when the container launches
CMD ["uvicorn", "app.main:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "8000"]

# For debuging purposes, uncomment the following lines
#COPY ./docker/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
#RUN apk add --no-cache bash
#ENTRYPOINT [ "docker-entrypoint.sh"]
#CMD ["bash"]