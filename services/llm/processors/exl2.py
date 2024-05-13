import os
import pysbd
import torch
from typing import List, Optional
from pydantic import BaseModel
from textwrap import dedent
from lmformatenforcer.integrations.exllamav2 import ExLlamaV2TokenEnforcerFilter
from lmformatenforcer import JsonSchemaParser
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Tokenizer,
)
from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler,
)
from exllamav2.generator.filters import (
    ExLlamaV2PrefixFilter
)
from config import LLMConfig

model_templates = {
    "dolphin-2.6-mixtral-8x7b-3.5bpw-h6-exl2": {
        'template': dedent(
            """<|im_start|>system
            {system_prompt}<|im_end|>
            <|im_start|>user
            {prompt}<|im_end|>
            <|im_start|>assistant
            {start_response}"""
        ),
        'eos_tag': '<|im_end|>'
    },
    "miqu-1-70b-sf-2.4bpw-h6-exl2": {
        'template': dedent(
            """[INST]system
            {system_prompt}[/INST]
            [INST]user
            {prompt}[/INST]
            [INST]assistant
            {start_response}"""
        ),
        'eos_tag': '[/INST]'
    }
}

class EXL2:

    def __init__(self, name, path, **kwargs):
        
        # Define and retrieve model if required
        self.model_name = name
        self.model_dir = path
        self.init_model_processor()
            
    
    # Setup initial config values     
    def init_model_processor(self):
        self.device = LLMConfig.DEVICE
        torch.cuda.set_device(self.device)
        self.debug_mode = LLMConfig.DEBUG
        self.seq_len = LLMConfig.SEQ_LEN
        self.exconfig = ExLlamaV2Config(self.model_dir + self.model_name)
        self.exconfig.prepare()
        self.exconfig.max_seq_len = self.seq_len
        self.model = None
        self.cache = None
        self.tokenizer = None
        self.generator = None
        self.seg = pysbd.Segmenter(language="en", clean=False)


    # Loads the faster whisper model
    def load_model(self):
        
        model = ExLlamaV2(self.exconfig)
        self.cache = ExLlamaV2Cache_Q4(model, lazy = True)
        model.load_autosplit(self.cache)
        self.tokenizer = ExLlamaV2Tokenizer(self.exconfig)
        self.generator = ExLlamaV2StreamingGenerator(model, self.cache, self.tokenizer)
        self.generator.set_stop_conditions([self.tokenizer.eos_token_id])
        self.generator.speculative_ngram = False
        self.generator.warmup()
        self.model = model


    # Send a prompt to the LLM processor
    def prompt(
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
            ):
        
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.top_p = top_a
        settings.token_repetition_penalty = repeat_penalty
        
        filters = []
        if guidance is not None:
            filters = self._processPydanticObject(guidance)
            settings.filters = filters
        settings.filter_prefer_eos = eos_bias
        input_text = model_templates[self.model_name]['template'].format(
                prompt=prompt, 
                system_prompt=system_prompt,
                start_response=response_prompt
        )

        input_ids = self.tokenizer.encode(input_text, add_bos=True)
        prompt_tokens = input_ids.shape[-1]
        self.generator.set_stop_conditions([self.tokenizer.eos_token_id])
        self.generator.begin_stream_ex(input_ids, settings)
        
        if self.debug_mode:
            print("--------------------------------------------------")
            print(prompt)
            print(" ------>" + (" (filtered)" if len(filters) > 0 else ""))

    # Request tokens from the LLM processor (after a prompt was supplied)
    def generate(self):
        response = ""
        while True:
            resp = self.generator.stream_ex()
            if resp["eos"] or model_templates[self.model_name]['eos_tag'] in response:
                return None
            else:
                yield resp["chunk"]
           

    def _processPydanticObject(self, obj: BaseModel):
        schema_parser = JsonSchemaParser(obj.schema())
        lmfe_filter = ExLlamaV2TokenEnforcerFilter(schema_parser, self.tokenizer)
        prefix_filter = ExLlamaV2PrefixFilter(self.model, self.tokenizer, "{")
        return [lmfe_filter, prefix_filter]
    