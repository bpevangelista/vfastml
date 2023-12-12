import argparse, uvicorn
from entrypoints import api_server
from engine.inference_engine import AsyncInferenceEngine, AsyncInferenceEngineParams


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    return args

def main():
    args = handle_args()

    api_server.inference_engine = AsyncInferenceEngine()
    uvicorn.run(
        api_server.app,
        host=args.host,
        port=args.port,
        log_level="debug",
        loop="uvloop",
    )


if __name__ == "__main__":
    main()
