import argparse
import json
import os
import pickle
import random
import subprocess
from dataclasses import dataclass


DATA_FOLDER_PATH: str = './data'
CACHED_CHAT_COMPLETIONS_PATH: str = os.path.join(DATA_FOLDER_PATH, 'varsize_chat_completions.bin')

@dataclass
class BenchmarkChatCompletionMessage:
    prompt: str
    generation_length: int


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


def read_json_file(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            json_data = json.load(file)
            return json_data
    except FileNotFoundError:
        print(f'File not found: {file_path}')
    except json.JSONDecodeError as err:
        print(f'Failed to parse JSON: {err}')


def gen_variable_size_chat_completions(file_path: str,
                                       max_seq_length: int) -> list[BenchmarkChatCompletionMessage]:

    json_dict = read_json_file(file_path)
    chat_completions: list[BenchmarkChatCompletionMessage] = []

    for chat in json_dict:
        if len(chat['conversations']) < 2:
            continue

        # Trying similar logic to vllm benchmark
        prompt = chat['conversations'][0]['value']
        reply = chat['conversations'][1]['value']
        full_seq_length = len(prompt) + len(reply)
        if full_seq_length > max_seq_length:
            continue

        chat_completion = BenchmarkChatCompletionMessage(prompt, full_seq_length)
        chat_completions.append(chat_completion)

    random.shuffle(chat_completions)
    return chat_completions


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_prompts', type=int, default=-1)
    parser.add_argument('--max_chat_length', type=int, default=1024)
    return parser.parse_args()


def main():
    args = handle_args()

    # Same dataset as vLLM
    url = ('https://huggingface.co/datasets/anon8231489123/' +
           'ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json')
    file_path = wget_url(url)

    chat_completions = gen_variable_size_chat_completions(file_path, args.max_chat_length)
    if args.max_prompts != -1:
        chat_completions = chat_completions[:args.max_prompts]
    save_cached_chat_completions(chat_completions)

main()