import argparse

import uvicorn

from kfastml.engine.inference_engine import AsyncInferenceEngine
from kfastml.entrypoints import api_server


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    return args


def api_main():
    args = handle_args()
    # args = read_args_from_yaml()

    #api_server = ApiServer()
    #api_server.run()

    api_server.inference_engine = AsyncInferenceEngine()

    uvicorn.run(
        api_server.app,
        host=args.host,
        port=args.port,
        log_level='debug',
        loop='uvloop',
    )


if __name__ == '__main__':
    api_main()
