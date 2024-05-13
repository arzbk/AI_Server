import bentoml
import torch
import whisperx
import time, threading, os, sys
from pathlib import Path
from config import STTConfig

LANGUAGE_CODE = "en"

@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 10},
)
class STTService:

    def __init__(self):
        self.cuda = torch.cuda.is_available()
        self.device = STTConfig.DEVICE
        self.model_name = STTConfig.MODEL_NAME
        torch.cuda.set_device(self.device)
        self.batch_size = 4 # reduce if low on GPU mem
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
        if self.cuda:
            compute_type = "float16"
            model = whisperx.load_model(self.model_name, device="cuda", device_index=self.device, compute_type=compute_type, language=LANGUAGE_CODE)
        else: 
            "int8"
            model = whisperx.load_model(self.model_name, device="cpu", device_index=self.device, compute_type=compute_type, language=LANGUAGE_CODE)   
        self.model_online = True
        self._start_activity_supervisor()
        return model


    # Unload the faster whisper model
    def unload_model(self):
        import gc; gc.collect(); torch.cuda.empty_cache(); del self.model
        self.model_online = False


    #Load Faster-Whisper model (consider specifying language and task options)
    @bentoml.api
    def transcribe(self, audio_file):
        self.last_activity = time.time()
        if not self.model_online: self.model = self.load_model()
        audio = whisperx.load_audio(audio_file)
        result = self.model.transcribe(audio, batch_size=self.batch_size)
        return {'text': " ".join([seg['text'] for seg in result['segments']])}

