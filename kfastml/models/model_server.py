import asyncio
from abc import abstractmethod, ABC

import torch
import uvloop
import zmq.asyncio

from kfastml.utils import DEFAULT_API_SERVER_RPC_PORT, DEFAULT_MODEL0_SERVER_RPC_PORT

# Use uvloop instead of asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class ModelServer(ABC):
    def __init__(self, model_type: str, model_uri: str, model_device: str = 'auto',
                 model_generation_params: dict = None,
                 api_rpc_port: int = DEFAULT_API_SERVER_RPC_PORT,
                 model_rpc_port: int = DEFAULT_MODEL0_SERVER_RPC_PORT,
                 ) -> None:
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

        # config torch
        torch.set_default_device(model_device)

    def _init_rpc(self):
        context = zmq.asyncio.Context(2)
        print(f'PULL tcp://127.0.0.1:{self.model_rpc_port}')
        print(f'PUSH tcp://127.0.0.1:{self.api_rpc_port}')

        self.pull_socket = context.socket(zmq.PULL)
        self.pull_socket.bind(f'tcp://127.0.0.1:{self.model_rpc_port}')
        self.push_socket = context.socket(zmq.PUSH)
        self.push_socket.connect(f'tcp://127.0.0.1:{self.api_rpc_port}')

    @abstractmethod
    async def _load_model(self):
        pass

    async def _rpc_loop(self):
        result = None
        while self._is_running:
            # debug
            print('rpc loop', result)

            result = await self.pull_socket.recv_json()
            request_id = result['request_id']
            self.push_socket.send_json({
                'request_id': request_id,
                'result': {
                    'result': 'blah'
                }
            })

    async def _main_loop(self):
        while self._is_running:
            # debug
            print('_main_loop')

            await asyncio.sleep(2)

    def run(self):
        assert not self._is_running

        self._is_running = True
        self.event_loop.create_task(self._rpc_loop())
        self.event_loop.create_task(self._main_loop())
        self.event_loop.run_forever()
