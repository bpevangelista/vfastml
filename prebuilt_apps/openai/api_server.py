import argparse
from typing import Literal

from pydantic import BaseModel

import kfastml.utils.api as api_utils
from kfastml.engine.dispatch_requests import TextGenerationReq, TextGenerationMessage
from kfastml.entrypoints.api_server import FastMLServer as ml_server


class ChatCompletionsMessage(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatCompletionsMessage] | str


@ml_server.app.post(path='/v1/chat/completions', status_code=200)
async def chat_completions(request: ChatCompletionsRequest):
    request_id = api_utils.gen_request_id('completions')

    available_models = {
        'llama2': {'model_uri': 'meta-llama/Llama-2-7b-chat-hf'},
        'mistral': {'model_uri': 'mistralai/Mistral-7B-v0.1'},
        'phi2': {'model_uri': 'microsoft/phi-2'},
    }
    selected_model = available_models.get(request.model, available_models['mistral'])

    messages = request.messages if isinstance(request.messages, str) else \
        [TextGenerationMessage(msg.role, msg.content) for msg in request.messages]

    dispatch_task = ml_server.dispatch_engine.dispatch(
        TextGenerationReq(
            request_id=request_id,
            model_uri=selected_model['model_uri'],
            model_adapter_uri=None,
            messages=messages))
    task_result = await dispatch_task.get_result()

    return api_utils.build_json_response(request_id, task_result)


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_level', type=str, default='debug')
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    return parser.parse_args()


if __name__ == '__main__':
    args = handle_args()

    ml_server.configure(log_level=args.log_level, dispatch_engine='async')
    ml_server.run(
        host=args.host,
        port=args.port,
    )
