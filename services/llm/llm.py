import os
from huggingface_hub import snapshot_download
from typing import AsyncGenerator
import asyncio
import pysbd
import bentoml
import annotated_types as AT
from typing_extensions import Annotated
import torch
from pydantic import BaseModel, conlist, dataclasses
from typing import Literal
from typing import List, Optional
from textwrap import dedent
import threading, time, json, sys
from .processors.exl2 import EXL2
from config import LLMConfig

@bentoml.service(
    resources={"gpu": 1},
    traffic={"timeout": 30}
)
class LLMService:

    def __init__(self):
        
        # Define and retrieve model if required
        self.model_name = LLMConfig.MODEL_NAME
        self.model_dir = LLMConfig.MODEL_DIR
        if not os.path.exists(self.model_dir + self.model_name):
            print("Downloading model: " + self.model_name)
            self._download_model(self.model_name)
            
        # Load appropriate LLM processor - EXL2 or VLLM
        if 'EXL' in self.model_name.upper() or 'GPTQ' in self.model_name.upper():
            self.processor = EXL2(self.model_name, self.model_dir)
        else:
            pass
            #self.processor = VLLM(self.model_name, self.model_dir)
        
        # Load the model with processor
        self.last_activity = time.time()
        self.processor.load_model()
        self.model_online = True
        self._start_activity_supervisor()
        
        # Conditionally segment output into sentences (helps for TTS)
        self.segment_sentences = LLMConfig.SEGMENT_SENTENCES
        if self.segment_sentences:
            self.segment = pysbd.Segmenter(language="en", clean=False)
        

    def _download_model(self, repo):
        snapshot_download(
            repo_id=repo, 
            revision=LLMConfig.REPO_BRANCH,
            local_dir=self.model_dir + self.model_name, 
            local_dir_use_symlinks=False
        )
        

    def _start_activity_supervisor(self):
        self.activity_thread = threading.Thread(target=self._check_model_uptime)
        self.activity_thread.daemon = True
        self.activity_thread.start()


    def _check_model_uptime(self):
        while self.model_online:
            current_time = time.time()
            elapsed_time = current_time - self.last_activity
            if elapsed_time > 1200:
                self.unload_model()
                break
            time.sleep(300) 
            
            
    # Unload the faster whisper model
    def unload_model(self):
        import gc; gc.collect(); torch.cuda.empty_cache(); del self.processor.model
        self.model_online = False


    #Load Faster-Whisper model (consider specifying language and task options)
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
        
        # Declare vars for method
        response = ""
        
        # Ensure model ready
        if not self.model_online: 
            self.processor.load_model()
        else:
            self.last_activity = time.time()
            
        # Prompt the model
        self.processor.prompt(
            prompt=prompt,
            system_prompt=system_prompt,
            response_prompt=response_prompt,
            guidance=guidance, 
            max_new_tokens=max_new_tokens, 
            temperature=temperature,
            eos_bias=eos_bias,
            repeat_penalty=repeat_penalty,
            top_k=top_k,
            top_p=top_p,
            top_a=top_a
        )
            
        # Group tokens by sentence and yield sentence at a time
        if self.segment_sentences:
            i, sntc_cnt = 0, 0
            while resp := self.processor.generate():
                response += resp["chunk"]
                if i % 5 == 0:
                    segs = self.seg.segment(response)
                    num_sentences = len(segs) - 1
                    if num_sentences > sntc_cnt:
                        sntc_cnt = num_sentences
                        yield segs[num_sentences - 1]
                        await asyncio.sleep(0)
            yield segs[num_sentences - 1]
            await asyncio.sleep(0)
        
        # ... or Yield generated tokens directly
        else:
            while resp := self.processor.generate():
                yield resp["chunk"]
                response += resp["chunk"]
                await asyncio.sleep(0)
    

# Test for LLM service
def llm_service_test(SERVER, PORT, ENDPOINT="llmservice", text="Write me a mean and cruel poem about amputees."):

    from pathlib import Path

    with bentoml.SyncHTTPClient(f"http://{SERVER}:{PORT}") as client:
    
        # Make the request using the Service endpoint
        resp_generator = client.generate_stream(
            prompt=text
        )
        
        response = ""
        for chunk in resp_generator:
            print(chunk, end="", flush=True)
            response += chunk
        
        print("")

        if response:
            return True
        else:
            return False
        
       
async def testLLMService():
    test = LLMService()
    resp = test.generate_stream(
        prompt="Write a song about a man named \"Jimbo\"",
        system_prompt=dedent("""You are a helpful assistant.""")
    )
    
    async for chunk in resp:
        print(chunk, flush=True, end="")       
    
        
if __name__ == "__main__":
    asyncio.run(testLLMService())
