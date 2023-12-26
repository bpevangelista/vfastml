import asyncio
from abc import abstractmethod, ABC

import torch
import zmq
import uvloop

# Use uvloop instead of asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class ModelServer(ABC):
    def __init__(self, server_port: int, server_type: str, model_uri: str, model_device: str, model_params: dict):
        self.server_type = server_type
        self.model_uri = model_uri
        self.model_device = model_device
        self.model_params = model_params

        self._is_running = False
        self._is_model_loaded = False
        self.event_loop = asyncio.new_event_loop()

        context = zmq.Context()
        self.rep_socket = context.socket(zmq.REP)
        self.rep_socket.bind(f'tcp://127.0.0.1:{server_port}')

        torch.set_default_device(model_device)
        self.event_loop.create_task(self._load_model())

    @abstractmethod
    async def _load_model(self):
        pass

    async def main_loop(self):
        while self._is_running:
            # debug
            print('loop')
            await asyncio.sleep(1)

    def run(self):
        assert not self._is_running

        self._is_running = True
        self.event_loop.create_task(self.main_loop())
        self.event_loop.run_forever()
