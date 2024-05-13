import bentoml
import io
import tempfile
import torch
from TTS.utils.manage import ModelManager
from TTS.api import TTS
import time, threading, os, sys
from pathlib import Path
import typing as t
import wave
from config import TTSConfig

@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 30},
)
class TTSService:

    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.device = TTSConfig.DEVICE
        torch.cuda.set_device(self.device)
        self.model_name = TTSConfig.MODEL_NAME
        self.model_manager = ModelManager()
        self.model_manager.download_model(self.model_name)
        self.model_online = False
        self.last_activity = time.time()
        self.model = self.load_model()


    def _start_activity_supervisor(self):
        self.activity_thread = threading.Thread(target=self._check_model_uptime)
        self.activity_thread.daemon = True
        self.activity_thread.start()


    def _check_model_uptime(self):
        while self.model_online:
            current_time = time.time()
            elapsed_time = current_time - self.last_activity
            if elapsed_time > 1200:  # 20 minutes = 1200 seconds
                self.unload_model()
                break
            time.sleep(300)  # Check every 5 minutes

                                  
    # Loads the faster whisper model
    def load_model(self):
        model = TTS(self.model_name)
        model.batch_size = 8
        model.to(f"cuda:{str(self.device)}")
        self.model_online = True
        self._start_activity_supervisor()
        return model


    # Unload the faster whisper model
    def unload_model(self):

        # Taken from here: https://github.com/m-bain/whisperX on the Python Usage example
        # OpenAI seem to know their stuff, so I'm going to trust them on this one
        import gc; gc.collect(); torch.cuda.empty_cache(); del self.model
        self.model_online = False


    #Load Faster-Whisper model (consider specifying language and task options)
    @bentoml.api
    def speak(
            self, 
            text: str,
            lang: str = "en"
    ): 
        self.last_activity = time.time()
        if not self.model_online: self.model = self.load_model()
        with tempfile.NamedTemporaryFile() as f:
            self.model.tts_to_file(text=text, file_path=f.name)
            f.seek(0)
            wav_data = f.read()
        return io.BytesIO(wav_data)
    
