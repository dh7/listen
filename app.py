from fastapi import FastAPI, File, UploadFile
import io

app = FastAPI()

@app.post("/hello/")
async def hello():
    return {"message": "Hello World"}

