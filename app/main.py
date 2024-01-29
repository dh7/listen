# Only for testing purposes
from fastapi import FastAPI
from typing import Union
import subprocess

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World!"}

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
    