import asyncio
from typing import Optional

import uvloop
import zmq.asyncio

from kfastml import log
from kfastml.engine.dispatch_requests import BaseDispatchEngineRequest, DispatchEngineRequestResult
from kfastml.utils import DEFAULT_API_SERVER_RPC_PORT, DEFAULT_MODEL0_SERVER_RPC_PORT

# Use uvloop instead of asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class DispatchEngineTaskResult:
    def __init__(self, finished_reason: str, result: dict):
        self.finished_reason = finished_reason
        self.result = result


class DispatchEngineTask:
    def __init__(self, request_id: str):
        self.request_id = request_id
        self._task_result = None
        self._finished = asyncio.Event()

    def _set_finished(self, finished_reason: str, result: dict):
        self._task_result = DispatchEngineTaskResult(finished_reason, result)
        self._finished.set()

    async def get_result(self) -> DispatchEngineTaskResult:
        await self._finished.wait()
        return self._task_result


class AsyncDispatchEngineParams:
    def __init__(self):
        pass


class AsyncDispatchEngine:
    FINISHED_REASON_DONE: str = 'done'
    FINISHED_REASON_CANCELED: str = 'canceled'
    FINISHED_REASON_ERROR: str = 'error'

    def __init__(self,
                 api_rpc_port: int = DEFAULT_API_SERVER_RPC_PORT,
                 model_rpc_port: int = DEFAULT_MODEL0_SERVER_RPC_PORT,
                 ):
        self.api_rpc_port = api_rpc_port
        self.model_rpc_port = model_rpc_port

        self._id_to_task_map = {}

        self._is_running = False
        self._event_loop = None
        self._init_rpc()

    def _init_rpc(self):
        context = zmq.asyncio.Context(2)
        log.info(f'PULL tcp://127.0.0.1:{self.api_rpc_port}')
        log.info(f'PUSH tcp://127.0.0.1:{self.model_rpc_port}')

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f'tcp://127.0.0.1:{self.api_rpc_port}')
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(f'tcp://127.0.0.1:{self.model_rpc_port}')

    async def _rpc_loop(self):
        while self._is_running:
            log.debug('_rpc_loop')

            req_result: DispatchEngineRequestResult = await self.pull_socket.recv_pyobj()
            assert isinstance(req_result, DispatchEngineRequestResult)

            self._id_to_task_map[req_result.request_id]._set_finished(
                req_result.finished_reason, req_result.result)

    async def _main_loop(self):
        log_count = 0
        while self._is_running:
            if log_count % 10 == 0:
                log.debug('_main_loop')
            log_count += 1

            await asyncio.sleep(1)

    def dispatch(self, request: BaseDispatchEngineRequest) -> DispatchEngineTask:
        assert request.request_id not in self._id_to_task_map, 'duplicate request_id. Internal error?'

        dispatch_task = DispatchEngineTask(request.request_id)
        self._id_to_task_map[request.request_id] = dispatch_task
        self.push_socket.send_pyobj(request)

        return dispatch_task

    def cancel(self, request_id: str) -> bool:
        assert False, 'Not Implemented'
        return False

    def run(self, event_loop: Optional[asyncio.AbstractEventLoop] = None):
        assert self._is_running is False, 'Already running. Are you starting it twice?'

        self._is_running = True
        self._event_loop = event_loop or asyncio.get_event_loop()

        self._event_loop.create_task(self._rpc_loop())
        self._event_loop.create_task(self._main_loop())
