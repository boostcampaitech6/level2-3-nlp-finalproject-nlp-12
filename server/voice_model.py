import whisper
from loguru import logger
import time
from dotenv import load_dotenv
import os
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
import pandas as pd

load_dotenv()


def load_whisper_pipeline():
    processer = WhisperProcessor.from_pretrained(os.getenv("MODEL_NAME"))
    model = WhisperForConditionalGeneration.from_pretrained(os.getenv("MODEL_NAME"))

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        feature_extractor=processer.feature_extractor,
        tokenizer=processer.tokenizer,
        chunk_length_s=30,
        device=os.getenv("DEVICE"),
    )

    logger.info("load_whisper_pipeline complete")
    
    return asr_pipe


def predict_whisper(pipe, audio_file):
    start = time.time()

    audio = whisper.load_audio(audio_file)
    prediction = pipe(audio.copy(), batch_size=8, return_timestamps=True, generate_kwargs={"language": "korean", "task": "transcribe"})["chunks"]

    result = ""
    for pred in prediction:
        result += pred["text"] + " "
    logger.info(f"STT Time taken: {time.time() - start}")

    # print the recognized text
    return result