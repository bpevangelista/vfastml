from abc import ABC
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from kfastml.models.model_server_text import TextGenerationModelServer
from kfastml.utils import DEFAULT_API_SERVER_RPC_PORT, DEFAULT_MODEL0_SERVER_RPC_PORT


class TextGenerationModelServerHF(TextGenerationModelServer, ABC):

    def __init__(self, model_type: str, model_uri: str, model_device: str = 'auto',
                 model_generation_params: dict = None, api_rpc_port: int = DEFAULT_API_SERVER_RPC_PORT,
                 model_rpc_port: int = DEFAULT_MODEL0_SERVER_RPC_PORT):
        super().__init__(model_type, model_uri, model_device, model_generation_params, api_rpc_port, model_rpc_port)
        self.config = None
        self.tokenizer = None
        self.model = None

    async def _load_model(self):
        self.config = AutoConfig.from_pretrained(
            self.model_uri,
            trust_remote_code=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_uri,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_uri,
            trust_remote_code=True,

            torch_dtype=torch.float16,  # Default f16 (TODO use b16 if supported)
            load_in_8bit=False,         # True if Training
            low_cpu_mem_usage=False,    # True for unified memory (e.g. M1/M2)
            attn_implementation='flash_attention_2'
        )

        self._is_model_loaded = True

    async def _generate_text(self, prompt: str, generation_params: Optional[dict] = None):
        generated_tokens = self.tokenizer(prompt, return_tensors="pt")
        input_ids = generated_tokens.input_ids
        attention_mask = generated_tokens.attention_mask

        # TODO use generation_params
        model_params = {
            'do_sample': True,
            'top_k': 30,
            'top_p': 0.90,
            'temperature': 0.8,
            'pad_token_id': self.tokenizer.eos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,

            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        output = self.model.generate(**model_params)
        output_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_texts[0]
