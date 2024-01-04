import asyncio
from abc import abstractmethod, ABC

import torch
import uvloop
import zmq.asyncio

from kfastml import log
from kfastml.engine.dispatch_engine import AsyncDispatchEngine
from kfastml.engine.dispatch_requests import BaseDispatchEngineRequest, DispatchEngineRequestResult, \
    TextGenerationReq, ImageToImageReq
from kfastml.utils import DEFAULT_API_SERVER_RPC_PORT, DEFAULT_MODEL0_SERVER_RPC_PORT

# Use uvloop instead of asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class ModelServer(ABC):
    def __init__(self,
                 model_type: str,
                 model_uri: str,
                 model_device: str = 'auto',
                 model_generation_params: dict = None,
                 api_rpc_port: int = DEFAULT_API_SERVER_RPC_PORT,
                 model_rpc_port: int = DEFAULT_MODEL0_SERVER_RPC_PORT,
                 ):
        self.model_type = model_type
        self.model_uri = model_uri
        self.model_device = model_device
        self.model_generation_params = model_generation_params

        self.api_rpc_port = api_rpc_port
        self.model_rpc_port = model_rpc_port

        self._is_running = False
        self._is_model_loaded = False
        self.event_loop = asyncio.new_event_loop()
        self.event_loop.create_task(self._load_model())

        self._init_rpc()
        self._init_torch()

    def _init_rpc(self):
        context = zmq.asyncio.Context(2)
        log.info(f'PULL tcp://127.0.0.1:{self.model_rpc_port}')
        log.info(f'PUSH tcp://127.0.0.1:{self.api_rpc_port}')

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f'tcp://127.0.0.1:{self.model_rpc_port}')
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(f'tcp://127.0.0.1:{self.api_rpc_port}')

    def _init_torch(self):
        torch.set_default_dtype(torch.float16)
        torch.set_default_device(self.model_device)

    @abstractmethod
    async def _load_model(self):
        assert False, 'Not Implemented'

    @abstractmethod
    async def _rpc_step(self, rpc_obj: BaseDispatchEngineRequest) -> any:
        assert False, 'Not Implemented'

    async def _rpc_loop(self):
        while self._is_running:
            log.debug(f'_rpc_loop')

            rpc_obj: BaseDispatchEngineRequest = await self.pull_socket.recv_pyobj()
            assert isinstance(rpc_obj, (TextGenerationReq, ImageToImageReq))

            result = None
            finished_reason = AsyncDispatchEngine.FINISHED_REASON_DONE
            try:
                result = await self._rpc_step(rpc_obj)
            except Exception as e:
                finished_reason = AsyncDispatchEngine.FINISHED_REASON_ERROR

            self.push_socket.send_pyobj(DispatchEngineRequestResult(
                request_id=rpc_obj.request_id,
                finished_reason=finished_reason,
                result=result,
            ))

    async def _main_loop(self):
        log_count = 0
        while self._is_running:
            if log_count % 10 == 0:
                log.debug('_main_loop')
            log_count += 1

            await asyncio.sleep(1)

    def run(self):
        assert not self._is_running

        self._is_running = True
        self.event_loop.create_task(self._rpc_loop())
        self.event_loop.create_task(self._main_loop())
        self.event_loop.run_forever()
