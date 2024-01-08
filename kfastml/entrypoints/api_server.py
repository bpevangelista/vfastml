from contextlib import asynccontextmanager
from typing import Literal

import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response

from kfastml import log
from kfastml.engine.dispatch_engine import AsyncDispatchEngine


def app_startup():
    # noinspection PyProtectedMember
    if FastMLServer._dispatch_engine_type == 'async':
        FastMLServer.dispatch_engine = AsyncDispatchEngine()
    FastMLServer.dispatch_engine.run()


def app_shutdown():
    pass


@asynccontextmanager
async def lifespan(_: FastAPI):
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
