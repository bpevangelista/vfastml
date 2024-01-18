from abc import ABC

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

from vfastml import log
from vfastml.errors import InvalidRequestError
from vfastml.models_servers.model_server_text import TextGenerationModelServer, TextGenerationMessage


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

    async def _generate_text(self, request_id: str, prompt: str | list[TextGenerationMessage], **kwargs) -> dict:
        try:
            # Llama2 ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L284
            model_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            log.debug(f'apply_chat_template {model_prompt}')
        except Exception as e:
            raise InvalidRequestError(e)

        generated_tokens = self.tokenizer(model_prompt, return_tensors="pt")
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
            'max_new_tokens': 64,

            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        log.debug(f'model.generate')
        output = self.model.generate(**model_params)
        output_texts = self.tokenizer.batch_decode(output, skip_special_tokens=False)
        log.debug(f'tokenizer.batch_decode {output_texts}')

        # TODO Properly handle & return OpenAI result
        return {
            'choices': [
                {
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': output_texts[0]
                    }
                }
            ]
        }
