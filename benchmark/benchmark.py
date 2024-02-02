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
    prompt_num_tokens: int
    reply_num_tokens: int


def load_cached_chat_completions():
    try:
        with open(CACHED_CHAT_COMPLETIONS_PATH, 'rb') as file:
            return pickle.load(file)
    except Exception as err:
        print(f'Failed to load completions dataset. Did you generate it first? Err: {err}')


def get_tokenizer(model_uri: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_uri,
        trust_remote_code = True,
        token = os.environ.get('HF_ACCESS_TOKEN'),
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def benchmark_vllm(args: any,
                   chat_completions: list[BenchmarkChatCompletionMessage],
                   ):
    tokenizer = get_tokenizer(args.model_uri)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model = args.model_uri,
        tokenizer = tokenizer.name_or_path,
        tokenizer_mode = 'slow',  # TODO Same across benchmarks. Change all to Fast?
        #quantization=quantization,
        tensor_parallel_size = args.num_gpus,
        dtype = args.dtype,
        max_model_len = DEFAULT_MAX_SEQ_LENGTH,
        trust_remote_code = True,
        #token=os.environ.get('HF_ACCESS_TOKEN'), # TODO Token is not supported. Use huggingface-cli login
    )

    # Same logic as vllm bench
    for chat in chat_completions:
        sampling_params = SamplingParams(
            n = args.num_generations,
            ignore_eos = True,
            use_beam_search = False,
            temperature = 1.0,
            top_p = 1.0,
            # Note vllm max_tokens is actually max_new_tokens
            max_tokens = chat.reply_num_tokens,
        )

        # noinspection PyPackageRequirements
        llm._add_request(
            prompt = chat.prompt,
            prompt_token_ids = None,
            sampling_params = sampling_params,
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

    torch_dtype = getattr(torch, args.dtype)
    model_server = TextGenerationModelServerHF(
        model_type = 'text_generation',
        model_uri = args.model_uri,
        model_device = 'cuda:0',
        model_load_kwargs = {
            'torch_dtype': torch_dtype,
            #'load_in_8bit': True,
            'token': os.environ.get('HF_ACCESS_TOKEN'),
        },
        model_forward_kwargs = TextGenerationForward(
            num_generations = args.num_generations,
            temperature = 1.0,
            top_p = 1.0,
        ),
        log_level = 'info',
    )

    # Same logic as vllm bench
    for chat in chat_completions:
        request = TextGenerationReq(
            request_id = 'foobar',
            messages = chat.prompt,
            forward_params = TextGenerationForward(
                min_tokens = chat.prompt_num_tokens + chat.reply_num_tokens,
                max_tokens = chat.prompt_num_tokens + chat.reply_num_tokens,
            ),
            model_uri = args.model_uri,
            model_adapter_uri = None,
        )
        # noinspection PyProtectedMember
        asyncio.run(model_server._requests_queue.put(request))

    print('Starting benchmark...')
    start_time = time.time()
    model_server.run_until_idle()
    end_time = time.time()
    elapsed_time_sec = end_time - start_time

    return elapsed_time_sec


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('benchmark', type=str, default='vfastml',
                        choices=['vfastml', 'vllm', 'hf'])

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
    elif args.benchmark == 'hf':
        raise NotImplementedError
    else:
        raise ValueError('benchmark must be one of [vfastml, vllm, hf]')

    # TODO Count the actual number of tokens generated
    total_tokens = sum(chat.reply_num_tokens for chat in chat_completions)
    avg_gen_tokens = total_tokens/num_chats
    tokens_per_sec = total_tokens/elapsed_time_sec

    print(f'Finished in {elapsed_time_sec:.2f}s. Results:')
    print(f'  {num_chats} chats, {avg_gen_tokens:.2f} avg tokens/chat, {tokens_per_sec:.2f} tokens/s')

main()