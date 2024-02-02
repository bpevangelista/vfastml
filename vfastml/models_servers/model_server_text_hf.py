from abc import ABC

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from transformers.utils import is_flash_attn_2_available

from vfastml import log
from vfastml.engine.dispatch_requests import TextGenerationForward, TextGenerationReq, DispatchRequestResult
from vfastml.errors import InvalidRequestError
from vfastml.models_servers.model_server_text import TextGenerationModelServer, ChatCompletionsResultChoice, \
    ChatCompletionsResultChoiceMessage, ChatCompletionsResult, ChatCompletionsResultUsage


class TextGenerationModelServerHF(TextGenerationModelServer, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tokenizer = None


    async def _load_model(self):
        tokenizer_load_kwargs = {
            'token': self.model_load_kwargs.get('token', None)
        }

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_uri,
            padding_side='left',
            trust_remote_code=True,
            **tokenizer_load_kwargs,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        merged_model_load_kwargs = {
            'torch_dtype': torch.float16,
            'attn_implementation': 'flash_attention_2' if is_flash_attn_2_available() else 'sdpa',
        }
        merged_model_load_kwargs.update(self.model_load_kwargs)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_uri,
            device_map = self.model_device,
            trust_remote_code = True,
            **merged_model_load_kwargs,
        )


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


    async def _generate_text(self, requests: list[TextGenerationReq]):

        valid_requests: list[TextGenerationReq] = []
        batch_prompts: list[str] = []

        for request in requests:
            try:
                # Llama2 ref: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L284
                model_prompt = self.tokenizer.apply_chat_template(request.messages, tokenize=False, add_generation_prompt=True)
                log.debug(f'apply_chat_template {model_prompt}')

                batch_prompts.append(model_prompt)
                valid_requests.append(request)

            except Exception as err:
                log.info(f'UserError {err}')
                result = DispatchRequestResult.from_exception(request.request_id, InvalidRequestError(err.__str__()))
                await self._results_queue.put(result)

        if not valid_requests:
            return

        tokens = self.tokenizer(batch_prompts, return_tensors='pt', padding=True)
        batch_input_ids = tokens.input_ids
        batch_attention_mask = tokens.attention_mask

        merged_forward_kwargs = {
            'do_sample': True,
        }
        merged_forward_kwargs.update(self._openai_to_hf(self.model_forward_kwargs))
        # TODO Support multiple forward parameters
        merged_forward_kwargs.update(self._openai_to_hf(valid_requests[0].forward_params))

        log.debug(f'model.generate')
        output = self.model.generate(
            **merged_forward_kwargs,
            return_dict_in_generate = True,
            pad_token_id = self.tokenizer.eos_token_id,
            eos_token_id = self.tokenizer.eos_token_id,
            input_ids = batch_input_ids,
            attention_mask = batch_attention_mask,
        )

        output_texts = self.tokenizer.batch_decode(output.sequences, skip_special_tokens=False)
        log.debug(f'tokenizer.batch_decode {output_texts}')

        output_index = 0
        for i, request in enumerate(valid_requests):
            choices = []

            for j in range(request.forward_params.num_generations):
                content = output_texts[output_index]
                choice = ChatCompletionsResultChoice(
                    finish_reason = 'stop',
                    index = j,
                    message = ChatCompletionsResultChoiceMessage(
                        role = 'assistant',
                        content = content[(len(batch_prompts[i]) - len('<s>') + 1):].strip(),
                    ),
                    logprobs = None,
                )
                choices.append(choice)
                output_index = output_index + 1

            prompt_tokens = batch_input_ids.shape[1]
            total_tokens = output.sequences.shape[0] * output.sequences.shape[1] # TODO Calculate non EOS tokens
            result = ChatCompletionsResult(
                choices = choices,
                model = self.model_uri,
                system_fingerprint = '',
                object = 'chat.completion',
                usage = ChatCompletionsResultUsage(
                    prompt_tokens = prompt_tokens,
                    completion_tokens = total_tokens - prompt_tokens,
                    total_tokens = total_tokens,
                )
            )

            dispatch_result = DispatchRequestResult(request.request_id, result=result.model_dump())
            await self._results_queue.put(dispatch_result)
