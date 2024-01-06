from abc import ABC

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from kfastml.models.model_server_text import TextGenerationModelServer


class TextGenerationModelServerHF(TextGenerationModelServer, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None

    async def _load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_uri,
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_uri,
            trust_remote_code=True,

            torch_dtype=torch.float16,  # Default f16 (TODO use b16 if supported)
            load_in_8bit=False,  # True if Training
            low_cpu_mem_usage=False,  # True for unified memory (e.g. M1/M2)
            # TODO Fallback to 'sdpa'?
            attn_implementation='flash_attention_2' if is_flash_attn_2_available() else None
        )

        self._is_model_loaded = True

    async def _generate_text(self, prompt: str, **kwargs) -> dict | None:
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
            'max_new_tokens': 1024,

            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        output = self.model.generate(**model_params)
        output_texts = self.tokenizer.batch_decode(output, skip_special_tokens=True)
        return output_texts[0]
