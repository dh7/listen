# Use an official Python runtime with CUDA as a parent image
FROM nvidia/cuda:11.0-base-ubuntu20.04

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Python
RUN apt-get update && apt-get install -y python3.8 python3-pip

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
