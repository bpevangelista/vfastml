import asyncio
import traceback
from abc import abstractmethod, ABC
from typing import Literal

import torch
import uvloop
import zmq.asyncio

from vfastml import log
from vfastml.engine.dispatch_requests import BaseDispatchRequest, DispatchRequestResult
from vfastml.utils import DEFAULT_API_SERVER_RPC_PORT, DEFAULT_MODEL0_SERVER_RPC_PORT
# noinspection PyProtectedMember
from vfastml.utils import _is_package_available

# Use uvloop instead of asyncio
asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())


class ModelServer(ABC):
    def __init__(self,
                 model_type: str,
                 model_uri: str,
                 model_device: str = 'auto',
                 model_load_kwargs: dict | None = None,
                 model_forward_kwargs: dict | None = None,

                 api_rpc_port: int = DEFAULT_API_SERVER_RPC_PORT,
                 model_rpc_port: int = DEFAULT_MODEL0_SERVER_RPC_PORT,
                 log_level: Literal['debug', 'info', 'warn', 'error'] = 'info',
                 ):
        self.model = None
        self.model_type = model_type
        self.model_uri = model_uri
        self.model_device = model_device
        self.model_load_kwargs = model_load_kwargs
        self.model_forward_kwargs = model_forward_kwargs

        self.api_rpc_port = api_rpc_port
        self.model_rpc_port = model_rpc_port

        self._requests_queue: list[BaseDispatchRequest] = []
        self._results_queue: list[DispatchRequestResult] = []
        self._has_requests = asyncio.Event()
        self._has_results = asyncio.Event()
        self._requests_queue_not_full = asyncio.Event()
        self._results_queue_not_full = asyncio.Event()
        self._requests_queue_not_full.set()
        self._results_queue_not_full.set()

        self._is_running = False
        self._is_model_loaded = False
        self._event_loop = asyncio.new_event_loop()

        log.set_level(log_level)
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
        # log.debug(torch.__config__.show()) # Too verbose

        log.info(
            f'Torch ({torch.__version__}), CPUs: {torch.get_num_threads()}, ' +
            f'CUDA_GPUs: {torch.cuda.device_count()}, ' +
            f'MPS_GPU: {torch.backends.mps.is_available()}')

        backends = {
            # 'CPU': torch.backends.cpu.get_cpu_capability(), # requires torch^=2.1.2
            'CU_DNN': torch.backends.cudnn.is_available(),
            'MKL_DNN': torch.backends.mkldnn.is_available(),
            'MKL': torch.backends.mkl.is_available(),
            'OpenMP': torch.backends.openmp.is_available(),
        }
        torch_backends = ''
        for key, value in backends.items():
            torch_backends += ', ' if len(torch_backends) > 0 else '  '
            torch_backends += f'{key} {value}'
        log.info(torch_backends)

        for i in range(torch.cuda.device_count()):
            cuda_device = torch.cuda.device(i)
            device_info = torch.cuda.get_device_properties(cuda_device)
            device_memory_in_gb = device_info.total_memory / (1024 * 1024 * 1024)
            flash_attn_2 = _is_package_available('flash_attn', '2.4.2')
            log.info(f'  {i:2d}: {device_info.name} {round(device_memory_in_gb, 2)}GB')
            log.info(f'      {{bf16: {torch.cuda.is_bf16_supported()}, flash_attn_2: {flash_attn_2}}}')

        torch.set_default_device(self.model_device)
        # TODO When it gets released, call torch.get_default_device to make sure it was set properly


    @abstractmethod
    async def _load_model(self):
        raise NotImplementedError


    @abstractmethod
    async def _rpc_step(self):
        raise NotImplementedError


    async def _send_rpc_result(self, socket_flags: int = 0) -> bool:
        # noinspection PyBroadException
        try:
            if self._results_queue:
                rpc_result = self._results_queue.pop()
                await self.push_socket.send_pyobj(rpc_result, flags=socket_flags)
                return True
        except Exception:
            # TODO Lost request, should we somehow retry?
            log.error(traceback.format_exc())
            return True

        return False

    async def _recv_rpc_request(self, socket_flags: int = 0) -> bool:
        # noinspection PyBroadException
        try:
            request: BaseDispatchRequest = await self.pull_socket.recv_pyobj(flags=socket_flags)
            self._requests_queue.append(request)
        except zmq.Again: # No more messages
            return False
        except Exception:
            # We can't reply because there's no request_id
            log.error(traceback.format_exc())
        return True


    async def _rpc_recv_loop(self):
        while self._is_running:
            log.debug(f'_rpc_recv_loop')
            await self._requests_queue_not_full.wait()

            # Get all requests
            if await self._recv_rpc_request(): # blocking
                while len(self._requests_queue) < 256 and await self._recv_rpc_request(zmq.NOBLOCK):
                    continue
                if len(self._requests_queue) == 256:
                    self._requests_queue_not_full.clear()
                self._has_requests.set()


    async def _rpc_send_loop(self):
        while self._is_running:
            log.debug(f'_rpc_send_loop')
            await self._has_results.wait()

            # Send all results
            while await self._send_rpc_result(zmq.NOBLOCK):
                continue
            self._has_results.clear()

            if len(self._results_queue) < 256:
                self._results_queue_not_full.set()


    async def _rpc_loop(self):
        while self._is_running:
            log.debug(f'_rpc_loop')

            await self._rpc_step()


    async def _is_alive_loop(self):
        log_count = 0
        while self._is_running:
            if log_count % 10 == 0:
                log.debug('_is_alive_loop')
            log_count += 1

            await asyncio.sleep(1)


    def run(self):
        assert not self._is_running

        self._is_running = True
        self._event_loop.create_task(self._load_model())
        self._event_loop.create_task(self._rpc_recv_loop())
        self._event_loop.create_task(self._rpc_send_loop())
        self._event_loop.create_task(self._rpc_loop())
        self._event_loop.create_task(self._is_alive_loop())
        self._event_loop.run_forever()
