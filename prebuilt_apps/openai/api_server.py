import argparse
from typing import Literal

from pydantic import BaseModel

import vfastml.utils.api as api_utils
from vfastml.engine.dispatch_requests import TextGenerationReq, TextGenerationMessage
from vfastml.entrypoints.api_server import FastMLServer as ml_server


class ChatCompletionsMessage(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: str


class ChatCompletionsRequest(BaseModel):
    model: str
    messages: list[ChatCompletionsMessage] | str
    frequency_penalty: float|None = 0.0
    max_tokens: int|None = 2048
    n: int|None = 1
    stream: bool|None = False
    temperature: int|None = 1.0
    top_p: int|None = 1.0
    user: str|None
    logit_bias: dict|None = None            # NotImplemented
    logprobs: bool|None = None              # NotImplemented
    top_logprobs: bool|None = None          # NotImplemented
    presence_penalty: int|None = None       # NotImplemented
    response_format: object|None = None     # NotImplemented
    seed: int|None = None                   # NotImplemented
    stop: str|list|None = None              # NotImplemented
    tools: list|None = None                 # NotImplemented
    tool_choice: str|object|None = None     # NotImplemented


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
