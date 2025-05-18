from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.stt import transcribe_speech_to_text
from app.llm import generate_response
from app.tts import transcribe_text_to_speech
import os
import tempfile
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("voice-assistant")

app = FastAPI(title="Voice AI Assistant API")

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; limit to specific origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Voice AI Assistant API is running"}

@app.post("/voice-chat")
async def voice_chat(request: Request, file: UploadFile = File(...)):
    """
    Process voice chat workflow:
    1. Receive audio file from frontend
    2. Convert speech to text using Whisper
    3. Generate response using Gemini
    4. Convert response to speech
    5. Return audio file
    """
    logger.info(f"Received request from {request.client.host} with file: {file.filename}")
    
    # Log request headers for debugging
    logger.info(f"Request headers: {request.headers}")
    
    # 1. Read uploaded file
    contents = await file.read()
    file_size = len(contents)
    logger.info(f"File received: {file_size} bytes")
    
    if not contents or file_size == 0:
        logger.error("Empty file received")
        return JSONResponse(
            status_code=400,
            content={"error": "Empty file"}
        )
    
    # Create a temp dir that won't be deleted immediately
    temp_dir = tempfile.mkdtemp()
    logger.info(f"Created temporary directory: {temp_dir}")
    
    # 2. Speech to text
    logger.info("Starting speech-to-text processing")
    transcript = transcribe_speech_to_text(contents, file_ext=os.path.splitext(file.filename)[1])
    if transcript.startswith("[ERROR]"):
        logger.error(f"STT error: {transcript}")
        return JSONResponse(
            status_code=500,
            content={"error": transcript}
        )
    
    logger.info(f"Transcribed: {transcript}")
    
    # 3. Generate response from LLM
    logger.info("Generating LLM response")
    response_text = generate_response(transcript)
    if response_text.startswith("[ERROR]"):
        logger.error(f"LLM error: {response_text}")
        return JSONResponse(
            status_code=500,
            content={"error": response_text}
        )
    
    logger.info(f"LLM Response: {response_text}")
    
    # 4. Text to speech
    logger.info("Starting text-to-speech processing")
    audio_path = transcribe_text_to_speech(response_text)
    if not audio_path or audio_path.startswith("[ERROR]"):
        logger.error(f"TTS error: {audio_path}")
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to generate speech: {audio_path}"}
        )
    
    # Ensure the file exists and is readable
    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found at: {audio_path}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Audio file not found at: {audio_path}"}
        )
    
    file_size = os.path.getsize(audio_path)
    logger.info(f"Audio file saved at: {audio_path} ({file_size} bytes)")
    
    # 5. Return audio file
    logger.info(f"Sending response file: {audio_path}")
    return FileResponse(
        path=audio_path,
        media_type="audio/wav",
        filename="response.wav"
    )
    
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
