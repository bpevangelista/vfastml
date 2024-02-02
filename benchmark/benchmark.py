import argparse, os, pickle, sys, time
import asyncio

from dataclasses import dataclass
from transformers import AutoTokenizer
import torch

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


def benchmark_vllm(args: any,
                   chat_completions: list[BenchmarkChatCompletionMessage],
                   ):
    tokenizer = get_tokenizer(args.model_uri)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model_uri,
        tokenizer=tokenizer.name_or_path,
        tokenizer_mode='slow',  # TODO Same across benchmarks, change all to Fast?
        #quantization=quantization,
        tensor_parallel_size=args.num_gpus,
        dtype = args.dtype,
        max_model_len=DEFAULT_MAX_SEQ_LENGTH,
        trust_remote_code=True,
        token=os.environ.get('HF_ACCESS_TOKEN'),
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

        llm.generate(
            prompts=chat.prompt,
            prompt_token_ids=None,
            sampling_params=sampling_params,
        )

    print('Starting benchmark...')
    start_time = time.time()
    # noinspection PyProtectedMember
    llm._run_engine(use_tqdm=True)
    end_time = time.time()
    elapsed_time_sec = end_time - start_time

    return elapsed_time_sec


def benchmark_vfastml(args: any,
                   chat_completions: list[BenchmarkChatCompletionMessage],
                   ):
    tokenizer = get_tokenizer(args.model_uri)

    sys.path.append('../')
    from vfastml.models_servers.model_server_text_hf import TextGenerationModelServerHF
    from vfastml.engine.dispatch_requests import TextGenerationForward
    from vfastml.engine.dispatch_requests import TextGenerationReq

    model_server = TextGenerationModelServerHF(
        model_type = 'text_generation',
        model_uri = args.model_uri,
        model_device = 'cuda:0',
        model_load_kwargs = {
            'torch_dtype': torch.float16,
            #'load_in_8bit': True,
            'token': os.environ.get('HF_ACCESS_TOKEN'),
        },
        model_forward_kwargs = TextGenerationForward(
            num_generations = args.num_generations,
            temperature=1.0,
            top_p=1.0,
        ),
        log_level='error',
    )

    # Same logic as vllm bench
    for chat in chat_completions:
        request = TextGenerationReq(
            request_id = 'foobar',
            messages = chat.prompt,
            forward_params = TextGenerationForward(
                min_tokens=chat.generation_length,
                max_tokens=chat.generation_length,
            ),
            model_uri = args.model_uri,
            model_adapter_uri = None,
        )
        # noinspection PyProtectedMember
        asyncio.run(model_server._requests_queue.put(request))

    print('Starting benchmark...')
    start_time = time.time()
    model_server.run()
    end_time = time.time()
    elapsed_time_sec = end_time - start_time

    return elapsed_time_sec


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', type=str, default='vfastml',
                        choices=['vllm', 'vfastml', 'hf'])

    parser.add_argument('--model-uri', type=str, default=DEFAULT_LLM_URI)
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'bfloat16', 'float32'])
    parser.add_argument('--attn-implementation', type=str, default='eager',
                        choices=['eager', 'sdpa', 'flash-attn'])
    parser.add_argument('--num-gpus', type=int, default=1)

    parser.add_argument('--num-generations', type=int, default=1)
    parser.add_argument('--max-prompts', type=int, default=100)
    return parser.parse_args()


def main():
    args = handle_args()
    chat_completions = load_cached_chat_completions()
    chat_completions = chat_completions[:args.max_prompts]
    num_chats = len(chat_completions)

    if args.benchmark == 'vllm':
        elapsed_time_sec = benchmark_vllm(args, chat_completions)
    elif args.benchmark == 'vfastml':
        elapsed_time_sec = benchmark_vfastml(args, chat_completions)
    else:
        raise NotImplementedError

    # TODO Count the actual number of tokens generated
    total_tokens = sum(chat.generation_length for chat in chat_completions)
    tokens_sec = total_tokens / elapsed_time_sec

    print('results')
    print(f'  {num_chats} chats, {total_tokens/num_chats} avg_completion_length, {tokens_sec} tokens/s')

main()