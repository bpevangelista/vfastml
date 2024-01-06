from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

from kfastml import log
from kfastml.engine.dispatch_engine import AsyncDispatchEngine
from kfastml.engine.dispatch_requests import TextGenerationReq
from kfastml.utils.api import build_json_response, gen_request_id


def app_startup():
    if FastMLServer._dispatch_engine_type == 'async':
        FastMLServer.dispatch_engine = AsyncDispatchEngine()
    FastMLServer.dispatch_engine.run()


def app_shutdown():
    pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_startup()
    yield
    app_shutdown()


class FastMLServer:
    app = FastAPI(lifespan=lifespan)
    dispatch_engine: AsyncDispatchEngine = None

    _is_running = False
    _log_level: Literal['debug', 'info', 'warn', 'error'] = 'info'
    _dispatch_engine_type: Literal['sync', 'async'] = 'async'

    @staticmethod
    def configure(
            log_level: Literal['debug', 'info', 'warn', 'error'] = 'info',
            dispatch_engine: Literal['async'] = 'async',
    ):
        assert not FastMLServer._is_running, "Cannot configure while running!"
        assert dispatch_engine == 'async', "Only async dispatch engine is currently supported"
        FastMLServer._log_level = log_level
        FastMLServer._dispatch_engine_type = dispatch_engine

        log.set_level(FastMLServer._log_level)


    @staticmethod
    def run(**kwargs):
        assert not FastMLServer._is_running, "Already running!"
        uvicorn.run(
            FastMLServer.app,
            log_level=FastMLServer._log_level,
            loop='uvloop',
            **kwargs,
        )


@FastMLServer.app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


class ChatCompletionsRequest(BaseModel):
    engine: str | None = None
    system: str | None = None
    prompt: str


@FastMLServer.app.post(path='/v1/chat/completions', status_code=200)
async def chat_completions(request: ChatCompletionsRequest):
    request_id = gen_request_id('completions')

    available_engines = {
        'llama2': {'model': 'meta-llama/Llama-2-7b-chat-hf'},
        'mistral': {'model': 'mistralai/Mistral-7B-v0.1'},
        'phi2': {'model': 'microsoft/phi-2'},
    }
    selected_engine = available_engines.get(request.engine, available_engines['mistral'])

    dispatch_task = FastMLServer.dispatch_engine.dispatch(
        TextGenerationReq(
            request_id=request_id,
            model_uri=selected_engine['model'],
            model_adapter_uri=None,
            prompt=request.prompt,
            extra_params=request.__dict__))
    task_result = await dispatch_task.get_result()

    response = build_json_response(request_id, task_result)
    return response
