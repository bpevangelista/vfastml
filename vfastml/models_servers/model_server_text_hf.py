from abc import ABC

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.utils import is_flash_attn_2_available

from vfastml import log
from vfastml.engine.dispatch_requests import TextGenerationForward
from vfastml.errors import InvalidRequestError
from vfastml.models_servers.model_server_text import TextGenerationModelServer, TextGenerationMessage, \
    ChatCompletionsResultChoice, ChatCompletionsResultChoiceMessage, ChatCompletionsResult, ChatCompletionsResultUsage


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

        merged_load_kwargs = {
            'torch_dtype': torch.float16,
            'attn_implementation': 'flash_attention_2' if is_flash_attn_2_available() else 'sdpa',
        }
        merged_load_kwargs.update(self.model_load_kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_uri,
            device_map = self.model_device,
            trust_remote_code = True,
            **merged_load_kwargs,
        )

        self._is_model_loaded = True

    def _openai_to_hf(self, forward_params: TextGenerationForward) -> dict:
        forward_dict = forward_params.__dict__
        vfast_to_hf = {
            # Rename
            'min_tokens': 'min_length',
            'max_tokens': 'max_length',
            'num_generations': 'num_return_sequences',
            'logit_bias': 'sequence_bias',
            # Keep Name
            'max_time': 'max_time',
            'output_scores': 'output_scores',
            'repetition_penalty': 'repetition_penalty',
            'seed': 'seed',
            'stream': 'stream',
            'use_cache': 'use_cache',
            'temperature': 'temperature',
            'top_p': 'top_p',
        }
        forward_dict = { vfast_to_hf.get(old_key): value
                         for old_key, value in forward_dict.items() if old_key in vfast_to_hf }

        if forward_dict.pop('stream'):
            forward_dict['streamer'] = TextStreamer(self.tokenizer)

        # TODO NotImplemented
        if 'seed' in forward_dict:
            forward_dict.pop('seed')

        if 'repetition_penalty' in forward_dict:
            penalty = forward_dict['repetition_penalty']
            forward_dict['repetition_penalty'] = penalty * 0.5 + 1.0 + 1e06

        return forward_dict


    async def _generate_text(self,
                             request_id: str,
                             prompt: str | list[TextGenerationMessage],
                             forward_params: TextGenerationForward) -> ChatCompletionsResult:
        try:
            # Llama2 ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L284
            model_prompt = self.tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            log.debug(f'apply_chat_template {model_prompt}')
        except Exception as err:
            raise InvalidRequestError from err

        tokens = self.tokenizer(model_prompt, return_tensors="pt", add_special_tokens=False, return_length=True)

        merged_forward_kwargs = {
            'do_sample': True,
        }
        merged_forward_kwargs.update(self.model_forward_kwargs)
        merged_forward_kwargs.update(self._openai_to_hf(forward_params))

        log.debug(f'model.generate')
        output = self.model.generate(
            **merged_forward_kwargs,
            return_dict_in_generate = True,
            pad_token_id = self.tokenizer.eos_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            input_ids = tokens.input_ids,
            attention_mask = tokens.attention_mask,
        )

        output_texts = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=True)
        log.debug(f'tokenizer.batch_decode {output_texts}')

        choices = []
        for i, output_text in enumerate(output_texts):
            choice = ChatCompletionsResultChoice(
                finish_reason = 'stop',
                index = i,
                message = ChatCompletionsResultChoiceMessage(
                    role = 'assistant',
                    content = output_text[(len(model_prompt) - len('<s>') + 1):].strip(),
                ),
                logprobs = None,
            )
            choices.append(choice)

        prompt_tokens = tokens.input_ids.shape[1]
        total_tokens = output.sequences.shape[0] * output.sequences.shape[1] # TODO Calculate non EOS tokens
        result = ChatCompletionsResult(
            choices = choices,
            model = '',
            system_fingerprint = '',
            object = 'chat.completion',
            usage = ChatCompletionsResultUsage(
                prompt_tokens = prompt_tokens,
                completion_tokens = total_tokens - prompt_tokens,
                total_tokens = total_tokens,
            )
        )

        return result
