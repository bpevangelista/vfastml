import argparse
import os
import pickle
from dataclasses import dataclass
from datetime import time

import torch
from transformers import AutoTokenizer

CACHED_CHAT_COMPLETIONS_PATH: str = './data/varsize_chat_completions.bin'
DEFAULT_LLM_URI = 'meta-llama/Llama-2-7b-chat-hf'
DEFAULT_DTYPE = torch.float16
DEFAULT_MAX_SEQ_LENGTH = 2048

@dataclass
class BenchmarkChatCompletionMessage:
    prompt: str
    generation_length: int


def load_cached_chat_completions():
    try:
        with open(CACHED_CHAT_COMPLETIONS_PATH, 'rb') as file:
            return pickle.load(file)
    except Exception as err:
        print(f'Failed to load completions dataset. Did you generate it first? Err: {err}')


def get_tokenizer(model_uri: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_uri,
        padding_side='left',
        trust_remote_code=True,
        token=os.environ.get('HF_ACCESS_TOKEN'),
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-uri', type=str, default=DEFAULT_LLM_URI)
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--attn-implementation', type=str, default='eager',
                        choices=['eager', 'sdpa', 'flash-attn'])
    parser.add_argument('--num-gpus', type=int, default=1)

    parser.add_argument('--num-generations', type=int, default=1)
    parser.add_argument('--max-prompts', type=int, default=1000)
    return parser.parse_args()


def benchmark_vllm(args: any,
                   chat_completions: list[BenchmarkChatCompletionMessage],
                   ):
    tokenizer = get_tokenizer(args.model_uri)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_uri,
        #tokenizer=tokenizer,
        tokenizer=tokenizer.name_or_path,
        #quantization=quantization,
        tensor_parallel_size=args.num_gpus,
        trust_remote_code=True,
        dtype = args.dtype,
        max_model_len=DEFAULT_MAX_SEQ_LENGTH,
    )

    # Same logic as vllm bench
    for chat in chat_completions:
        sampling_params = SamplingParams(
            n=args.num_generations,
            temperature=1.0,
            top_p=1.0,
            use_beam_search=False,
            ignore_eos=True,
            max_tokens=chat.generation_length,
        )

        llm._add_request(
            prompt=chat.prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    startTime = time.time()
    llm._run_engine()
    endTime = time.time()
    elapsedTimeSec = endTime - startTime

    return elapsedTimeSec


def main():
    args = handle_args()
    chat_completions = load_cached_chat_completions()
    chat_completions = chat_completions[:args.max_prompts]

    elapsedTimeSec = benchmark_vllm(args, chat_completions)

    num_chats = len(chat_completions)
    total_tokens = sum(chat.generation_length for chat in chat_completions)
    tokens_sec = total_tokens / elapsedTimeSec

    print(f'{num_chats} chats, {total_tokens/num_chats} avg_completion_length, {tokens_sec} tokens/s')

main()