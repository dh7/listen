# Use an official Python runtime with CUDA as a parent image
FROM python:3.8-alpine

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Install pyannote.audio, FastAPI, Uvicorn, Whisper, and other dependencies
RUN pip3 install --no-cache-dir pyannote.audio fastai fastapi uvicorn whisper

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME SpeakerDiarization

# Run app.py with Uvicorn when the container launches
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
