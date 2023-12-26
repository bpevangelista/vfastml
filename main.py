import argparse
import uvicorn

from kfastml.engine.inference_engine import AsyncInferenceEngine
from kfastml.entrypoints import api_server
from kfastml.models.model_server_hf import ModelServerHF


def handle_api_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    return args


def handle_model_args():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    return args


def api_main():
    args = handle_api_args()
    # args = read_args_from_yaml()

    #api_server = ApiServer()
    #api_server.run()

    api_server.inference_engine = AsyncInferenceEngine()
    uvicorn.run(
        api_server.app,
        host=args.host,
        port=args.port,
        log_level="debug",
        loop="uvloop",
    )


def model_main():
    args = handle_model_args()
    # args = read_args_from_yaml()

    model_server = ModelServerHF(
        server_port=6410,
        server_type='text-generation',
        model_uri='mistralai/Mistral-7B-v0.1',
        model_device='cuda:0',
        model_params={
            'top_k': 30,
            'top_p': 0.9,
        }
    )
    model_server.run()


if __name__ == "__main__":
    #api_main()
    model_main()
