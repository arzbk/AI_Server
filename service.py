import io
from typing import Optional
import bentoml
from textwrap import dedent
from bentoml.io import File
from fastapi import FastAPI, WebSocket
import time, threading, os, sys
from pydantic import BaseModel
import torch
from pathlib import Path
import typing as t
from utils.console import *
from services.tts.tts import TTSService
from services.stt.stt import STTService
from services.llm.llm import LLMService
from typing import AsyncGenerator
import asyncio

LANGUAGE_CODE = "en"

# Create a FastAPI app instance
app = FastAPI()

@bentoml.service(
    resources={
        "cpu": "1"
    },
    traffic={"timeout": 30}
)
class AIService:
    
    llm = bentoml.depends(LLMService)
    stt = bentoml.depends(STTService)
    tts = bentoml.depends(TTSService)

    # TTS Endpoint
    @bentoml.api
    def speak(
            self, 
            text: str,
            lang: str = "en"
    ) -> t.Annotated[Path, bentoml.validators.ContentType('audio/*')]:
        res = self.tts.speak(
            text=text,
            lang=lang
        )
        print("\n\n - TTS ==> "+str(res))
        return Path(res)

    # STT Endpoint
    @bentoml.api
    def transcribe(
            self, 
            audio: Path,
    ):
        res = self.stt.transcribe(
            audio=audio
        )['text']
        print("\n\n - STT ==> "+str(res))
        return res
    
    # LLM Endpoint
    @bentoml.api
    async def generate_stream(
            self, 
            prompt: str,
            system_prompt: str = "You are a helpful assistant.",
            response_prompt: str = "",
            guidance: Optional[BaseModel] = None, 
            max_new_tokens: int = 2000, 
            temperature: float = 0.8,
            eos_bias: bool = True,
            repeat_penalty: float = 1.05,
            top_k: int = 50,
            top_p: int = 0.8,
            top_a: int = 0.0
    ) -> AsyncGenerator[str, None]:
        return self.llm.generate_stream(
            prompt,
            system_prompt,
            response_prompt,
            guidance, 
            max_new_tokens, 
            temperature,
            eos_bias,
            repeat_penalty,
            top_k,
            top_p,
            top_a)
            
    # Exposes end-to-end pipeline (Audio -> LLM -> Audio)
    @app.websocket("/converse")
    async def websocket_endpoint(self, ws: WebSocket):
        await ws.accept()
        while True:
            wav_data = await ws.receive_bytes()
            text = self.stt.transcribe(audio=io.BytesIO(wav_data))['text']
            async for text in self.llm.generate_stream(
                    prompt=text
                ):
                asyncio.create_task(self.send_tts_when_ready(ws, text))
                
    
    # Used to send TTS audio data to the client          
    async def send_tts_when_ready(self, ws, text):
        wav_data = self.tts.speak(text=text)
        ws.send_bytes()
        