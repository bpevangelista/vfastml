import argparse
import os

import torch

from vfastml.models_servers.model_server_text_hf import TextGenerationModelServerHF


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_device', type=str, default='cuda')
    parser.add_argument('--log_level', type=str, default='debug')
    return parser.parse_args()


if __name__ == '__main__':
    args = handle_args()

    torch.set_default_dtype(torch.float16)

    model_server = TextGenerationModelServerHF(
        model_type = 'text_generation',
        model_uri='mistralai/Mistral-7B-v0.1',
        #model_uri='meta-llama/Llama-2-7b-chat-hf',
        model_device = args.model_device,
        model_load_kwargs = {
            'torch_dtype': torch.float16,
            'load_in_8bit': True,
            'token': os.environ.get('HF_ACCESS_TOKEN'),
        },
        model_forward_kwargs = {
            'top_p': 0.9,
        },
        log_level=args.log_level,
    )
    model_server.run()
