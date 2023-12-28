import asyncio
from typing import Optional

import uvloop
import zmq.asyncio

from kfastml.utils import DEFAULT_API_SERVER_RPC_PORT, DEFAULT_MODEL0_SERVER_RPC_PORT

# Use uvloop instead of asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class InferenceTaskResult:
    def __init__(self, finished_reason: str, result: dict):
        self.finished_reason = finished_reason
        self.result = result


class InferenceTask:
    def __init__(self):
        self._task_result = None
        self._finished = asyncio.Event()

    def _set_finished(self, finished_reason: str, result: dict):
        self._task_result = InferenceTaskResult(finished_reason, result)
        self._finished.set()

    async def get_result(self) -> InferenceTaskResult:
        await self._finished.wait()
        return self._task_result


class AsyncInferenceEngineParams:
    def __init__(self):
        pass


class AsyncInferenceEngine:

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
        print(f'PULL tcp://127.0.0.1:{self.api_rpc_port}')
        print(f'PUSH tcp://127.0.0.1:{self.model_rpc_port}')

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f'tcp://127.0.0.1:{self.api_rpc_port}')
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(f'tcp://127.0.0.1:{self.model_rpc_port}')

    async def _rpc_loop(self) -> None:
        while self._is_running:
            # debug
            print('_rpc_loop')

            message = await self.pull_socket.recv_json()
            request_id = message['request_id']
            result = message['result']

            self._id_to_task_map[request_id]._set_finished(
                AsyncInferenceEngine.FINISHED_REASON_DONE, result)

    async def _main_loop(self) -> None:
        while self._is_running:
            # debug
            print('_main_loop')

            await asyncio.sleep(2)

    def dispatch(self, request_id: str, model_uri: str, model_adapter_uri: Optional[str] = None,
                 input_params: dict = None) -> InferenceTask:
        assert type(request_id) is str
        assert request_id not in self._id_to_task_map, 'duplicate request_id. Internal error?'

        self.push_socket.send_json({
            'request_id': request_id,
            'model_uri': model_uri,
            'model_adapter_uri': model_adapter_uri,
            'input_params': input_params
        })

        inference_task = InferenceTask()
        self._id_to_task_map[request_id] = inference_task

        return inference_task

    def cancel(self, request_id: str) -> bool:
        pass

    def run(self, event_loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        assert self._is_running is False, 'Already running. Are you starting it twice?'

        self._is_running = True
        self._event_loop = event_loop or asyncio.get_event_loop()

        self._event_loop.create_task(self._rpc_loop())
        self._event_loop.create_task(self._main_loop())
