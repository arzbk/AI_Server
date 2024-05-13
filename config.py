class LLMConfig:
    MODEL_NAME: str = "turboderp/Phi-3-mini-128k-instruct-exl2"
    MODEL_DIR: str = "/home/raisbecka/Projects/LLM_API/models/"
    REPO_BRANCH: str = "4.0bpw"
    SEGMENT_SENTENCES: bool = True
    DEVICE: int = 0
    DEBUG: bool = False
    SEQ_LEN: int = 2000
    

class TTSConfig:
    MODEL_NAME: str = "tts_models/en/ljspeech/glow-tts"
    DEVICE: int = 1
    COQUI_TOS_AGREED: bool = True
    

class STTConfig:
    MODEL_NAME: str = "large-v3"
    DEVICE: int = 1