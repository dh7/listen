# Only for testing purposes
from fastapi import FastAPI
from typing import Union
import subprocess
import os
import tempfile
import time

import numpy as np
import torch
import yt_dlp as youtube_dl
from gradio_client import Client
from pyannote.audio import Pipeline
from transformers.pipelines.audio_utils import ffmpeg_read

app = FastAPI()

YT_LENGTH_LIMIT_S = 36000  # limit to 10 hour YouTube files
SAMPLING_RATE = 16000

API_URL = "https://sanchit-gandhi-whisper-jax.hf.space/"

def load_HF_token(file_path):
    try:
        with open(file_path, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        print(f"Error: {file_path} not found.")
        return os.environ.get("HF_TOKEN")

HF_TOKEN = load_HF_token('secret-hf-token.txt')

# set up the Gradio client
client = Client(API_URL)

# set up the diarization pipeline
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=HF_TOKEN)


def format_string(timestamp):
    """
    Reformat a timestamp string from (HH:)MM:SS to float seconds. Note that the hour column
    is optional, and is appended within the function if not input.
    Args:
        timestamp (str):
            Timestamp in string format, either MM:SS or HH:MM:SS.
    Returns:
        seconds (float):
            Total seconds corresponding to the input timestamp.
    """
    split_time = timestamp.split(":")
    split_time = [float(sub_time) for sub_time in split_time]

    if len(split_time) == 2:
        split_time.insert(0, 0)

    seconds = split_time[0] * 3600 + split_time[1] * 60 + split_time[2]
    return seconds


# Adapted from https://github.com/openai/whisper/blob/c09a7ae299c4c34c5839a76380ae407e7d785914/whisper/utils.py#L50
def format_timestamp(seconds: float, always_include_hours: bool = False, decimal_marker: str = "."):
    """
    Reformat a timestamp from a float of seconds to a string in format (HH:)MM:SS. Note that the hour
    column is optional, and is appended in the function if the number of hours > 0.
    Args:
        seconds (float):
            Total seconds corresponding to the input timestamp.
    Returns:
        timestamp (str):
            Timestamp in string format, either MM:SS or HH:MM:SS.
    """
    if seconds is not None:
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
    else:
        # we have a malformed timestamp so just return it as is
        return seconds


def format_as_transcription(raw_segments):
    return "\n\n".join(
        [
            f"{chunk['speaker']} [{format_timestamp(chunk['timestamp'][0])} -> {format_timestamp(chunk['timestamp'][1])}] {chunk['text']}"
            for chunk in raw_segments
        ]
    )


def _return_yt_html_embed(yt_url):
    video_id = yt_url.split("?v=")[-1]
    HTML_str = (
        f'<center> <iframe width="500" height="320" src="https://www.youtube.com/embed/{video_id}"> </iframe>'
        " </center>"
    )
    return HTML_str


def download_yt_audio(yt_url, filename):
    info_loader = youtube_dl.YoutubeDL()
    try:
        info = info_loader.extract_info(yt_url, download=False)
    except youtube_dl.utils.DownloadError as err:
        raise str(err)

    file_length = info["duration_string"]
    file_length_s = format_string(file_length)

    if file_length_s > YT_LENGTH_LIMIT_S:
        yt_length_limit_hms = time.strftime("%HH:%MM:%SS", time.gmtime(YT_LENGTH_LIMIT_S))
        file_length_hms = time.strftime("%HH:%MM:%SS", time.gmtime(file_length_s))
        raise f"To encourage fair usage of the demo, the maximum YouTube length is {yt_length_limit_hms} got {file_length_hms} YouTube video."

    ydl_opts = {"outtmpl": filename, "format": "worstvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best"}
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        try:
            ydl.download([yt_url])
        except youtube_dl.utils.ExtractorError as err:
            raise gr.Error(str(err))


def align(transcription, segments, group_by_speaker=True):
    transcription_split = transcription.split("\n")

    # re-format transcription from string to List[Dict]
    transcript = []
    for chunk in transcription_split:
        start_end, transcription = chunk[1:].split("] ")
        start, end = start_end.split("->")

        transcript.append({"timestamp": (format_string(start), format_string(end)), "text": transcription})

    # diarizer output may contain consecutive segments from the same speaker (e.g. {(0 -> 1, speaker_1), (1 -> 1.5, speaker_1), ...})
    # we combine these segments to give overall timestamps for each speaker's turn (e.g. {(0 -> 1.5, speaker_1), ...})
    new_segments = []
    prev_segment = cur_segment = segments[0]

    for i in range(1, len(segments)):
        cur_segment = segments[i]

        # check if we have changed speaker ("label")
        if cur_segment["label"] != prev_segment["label"] and i < len(segments):
            # add the start/end times for the super-segment to the new list
            new_segments.append(
                {
                    "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["start"]},
                    "speaker": prev_segment["label"],
                }
            )
            prev_segment = segments[i]

    # add the last segment(s) if there was no speaker change
    new_segments.append(
        {
            "segment": {"start": prev_segment["segment"]["start"], "end": cur_segment["segment"]["end"]},
            "speaker": prev_segment["label"],
        }
    )

    # get the end timestamps for each chunk from the ASR output
    end_timestamps = np.array([chunk["timestamp"][-1] for chunk in transcript])
    segmented_preds = []

    # align the diarizer timestamps and the ASR timestamps
    for segment in new_segments:
        # get the diarizer end timestamp
        end_time = segment["segment"]["end"]
        # find the ASR end timestamp that is closest to the diarizer's end timestamp and cut the transcript to here
        upto_idx = np.argmin(np.abs(end_timestamps - end_time))

        if group_by_speaker:
            segmented_preds.append(
                {
                    "speaker": segment["speaker"],
                    "text": "".join([chunk["text"] for chunk in transcript[: upto_idx + 1]]),
                    "timestamp": (transcript[0]["timestamp"][0], transcript[upto_idx]["timestamp"][1]),
                }
            )
        else:
            for i in range(upto_idx + 1):
                segmented_preds.append({"speaker": segment["speaker"], **transcript[i]})

        # crop the transcripts and timestamp lists according to the latest timestamp (for faster argmin)
        transcript = transcript[upto_idx + 1 :]
        end_timestamps = end_timestamps[upto_idx + 1 :]

    # final post-processing
    transcription = format_as_transcription(segmented_preds)
    return transcription


def transcribe(audio_path, task="transcribe", group_by_speaker=True):
    # run Whisper JAX asynchronously using Gradio client (endpoint)
    job = client.submit(
        audio_path,
        task,
        True,
        api_name="/predict_1",
    )

    # run diarization while we wait for Whisper JAX
    progress(0, desc="Diarizing...")
    diarization = diarization_pipeline(audio_path)
    segments = diarization.for_json()["content"]

    # only fetch the transcription result after performing diarization
    progress(0.33, desc="Transcribing...")
    transcription, _ = job.result()

    # align the ASR transcriptions and diarization timestamps
    progress(0.66, desc="Aligning...")
    transcription = align(transcription, segments, group_by_speaker=group_by_speaker)

    return transcription


def transcribe_yt(yt_url, group_by_speaker=True):
    # run Whisper JAX asynchronously using Gradio client (endpoint)
    print("transcribe yt_url", yt_url)
    job = client.submit(
        yt_url,
        "transcribe",
        True,
        api_name="/predict_2",
    )
    print("transcribe yt_url done")
    html_embed_str = _return_yt_html_embed(yt_url)

    # download the YouTube video to a temporary directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        filepath = os.path.join(tmpdirname, "video.mp4")
        download_yt_audio(yt_url, filepath)
        with open(filepath, "rb") as f:
            inputs = f.read()

    inputs = ffmpeg_read(inputs, SAMPLING_RATE)
    inputs = torch.from_numpy(inputs).float()
    inputs = inputs.unsqueeze(0)

    # run diarization while we wait for Whisper JAX
    print("Diarizing...")
    diarization = diarization_pipeline(
        {"waveform": inputs, "sample_rate": SAMPLING_RATE},
    )
    segments = diarization.for_json()["content"]

    # only fetch the transcription result after performing diarization
    print("Transcribing...")
    _, transcription, _ = job.result()

    # align the ASR transcriptions and diarization timestamps
    print("Aligning...")
    transcription = align(transcription, segments, group_by_speaker=group_by_speaker)

    return html_embed_str, transcription


@app.get("/transcribe_yt/{yt_url}")
def run_transcribe_yt(yt_url: str):
    try:
        transcription = transcribe_yt(yt_url)

        return {"result": transcription}
    except Exception as e:
        return {"error": str(e)}

@app.get("/transcribe_test/")
def run_transcribe_test():
    try:
        transcription = transcribe_yt("https://www.youtube.com/watch?v=m4nQAs9oA_M")

        return {"result": transcription}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"Hello": "World today!"}

@app.get("/nvidia-smi")
def run_nvidia_smi():
    try:
        process = subprocess.Popen(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            return {"error": stderr.decode()}
        return {"result": stdout.decode()}
    except Exception as e:
        return {"error": str(e)}
    