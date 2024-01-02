from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from kfastml.engine.dispatch_engine import AsyncDispatchEngine
from kfastml.engine.dispatch_requests import TextGenerationReq, ImageToImageReq
from kfastml.utils.api import build_json_response, gen_request_id


def app_startup():
    dispatch_engine.run()


def app_shutdown():
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_startup()
    yield
    app_shutdown()


app = FastAPI(lifespan=lifespan)
dispatch_engine: Optional[AsyncDispatchEngine] = None


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


class ChatCompletionsRequest(BaseModel):
    engine: Optional[str] = None
    system: Optional[str] = None
    prompt: str


@app.post(path='/v1/chat/completions', status_code=200)
async def chat_completions(request: ChatCompletionsRequest):
    request_id = gen_request_id('completions')

    supported_engines = {
        'llama2': {
            'model': 'meta-llama/Llama-2-7b-chat-hf'
        },
        'mistral': {
            'model': 'mistralai/Mistral-7B-v0.1'
        },
        'phi2': {
            'model': 'microsoft/phi-2'
        },
    }

    if request.engine in supported_engines:
        desired_engine = supported_engines[request.engine]
    else:
        desired_engine = supported_engines['mistral']  # default

    dispatch_task = dispatch_engine.dispatch(
        TextGenerationReq(
            request_id=request_id,
            model_uri=desired_engine['model'],
            model_adapter_uri=None,
            prompt=request.prompt,
            extra_params=request.__dict__))
    task_result = await dispatch_task.get_result()

    response = build_json_response(request_id, task_result)
    return response


class ImageCleanupRequest(BaseModel):
    images_url: list[str]


@app.post(path='/v1/image/cleanup', status_code=200)
async def image_cleanup(request: ImageCleanupRequest):
    request_id = gen_request_id('completions')

    dispatch_task = dispatch_engine.dispatch(
        ImageToImageReq(request_id=request_id,
                        model_uri='wdnet/wdnet-1',
                        image_data=request.image_data,
                        extra_params=request.__dict__))
    task_result = await dispatch_task.get_result()

    return build_json_response(request_id, task_result)
