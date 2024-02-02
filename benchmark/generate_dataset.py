import argparse
import json
import os
import pickle
import random
import subprocess
import time
from dataclasses import dataclass

from transformers import AutoTokenizer

DEFAULT_LLM_URI = 'meta-llama/Llama-2-7b-chat-hf'
DATA_FOLDER_PATH: str = './data'
CACHED_CHAT_COMPLETIONS_PATH: str = os.path.join(DATA_FOLDER_PATH, 'varsize_chat_completions.bin')

@dataclass
class BenchmarkChatCompletionMessage:
    prompt: str
    prompt_num_tokens: int
    reply_num_tokens: int


def load_cached_chat_completions():
    if os.path.exists(CACHED_CHAT_COMPLETIONS_PATH):
        with open(CACHED_CHAT_COMPLETIONS_PATH, 'rb') as file:
            return pickle.load(file)


def save_cached_chat_completions(completions: list[BenchmarkChatCompletionMessage]):
    try:
        with open(CACHED_CHAT_COMPLETIONS_PATH, 'wb') as file:
            pickle.dump(completions, file)
    except Exception as err:
        print(f'Failed to write cached completions: {err}')


def wget_url(url: str):
    file_path = os.path.join(DATA_FOLDER_PATH, os.path.basename(url))
    if not os.path.exists(file_path):
        try:
            subprocess.run(['wget', url, '-P', DATA_FOLDER_PATH])
        except Exception as err:
            print(f'Failed to wget url: {err}')

    return file_path


def read_json_file(file_path: str) -> any:
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            return json_data
    except FileNotFoundError:
        print(f'File not found: {file_path}')
    except json.JSONDecodeError as err:
        print(f'Failed to parse JSON: {err}')


def gen_variable_size_chat_completions(file_path: str,
                                       args: any) -> list[BenchmarkChatCompletionMessage]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_uri,
        trust_remote_code=True,
        token=os.environ.get('HF_ACCESS_TOKEN'),
    )
    tokenizer.pad_token = tokenizer.eos_token

    json_blob: list[dict] = read_json_file(file_path)
    random.shuffle(json_blob)
    chat_completions: list[BenchmarkChatCompletionMessage] = []

    for chat in json_blob:
        if len(chat['conversations']) < 2:
            continue

        # Trying similar logic to vllm benchmark
        prompt = chat['conversations'][0]['value']
        reply = chat['conversations'][1]['value']
        prompt_tokens = len(tokenizer.tokenize(prompt, padding=False))
        reply_tokens = len(tokenizer.tokenize(reply, padding=False))

        total_tokens = prompt_tokens + reply_tokens
        if total_tokens > args.max_tokens:
            continue

        chat_completion = BenchmarkChatCompletionMessage(prompt, prompt_tokens, reply_tokens)
        chat_completions.append(chat_completion)
        if args.max_prompts != -1 and len(chat_completions) >= args.max_prompts:
            break

    return chat_completions


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-uri', type=str, default=DEFAULT_LLM_URI)
    parser.add_argument('--max-prompts', type=int, default=-1)
    parser.add_argument('--max-tokens', type=int, default=1024)
    return parser.parse_args()


def main():
    args = handle_args()

    # Same dataset as vLLM
    start_time = time.time()
    url = ('https://huggingface.co/datasets/anon8231489123/' +
           'ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json')
    file_path = wget_url(url)

    chat_completions = gen_variable_size_chat_completions(file_path, args)
    end_time = time.time()
    elapsed_time_sec = end_time - start_time
    print(f'Finished in {elapsed_time_sec:.2f}s. {len(chat_completions)} chat completions.')

    if args.max_prompts != -1:
        chat_completions = chat_completions[:args.max_prompts]
    save_cached_chat_completions(chat_completions)

main()