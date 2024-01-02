from abc import ABC, abstractmethod
from typing import Optional

from kfastml.engine.dispatch_requests import TextGenerationReq, ImageToImageReq
from kfastml.models.model_server import ModelServer
from kfastml.utils import DEFAULT_API_SERVER_RPC_PORT, DEFAULT_MODEL0_SERVER_RPC_PORT


class ImageToImageModelServer(ModelServer, ABC):

    def __init__(self,
                 model_type: str,
                 model_uri: str,
                 model_device: str = 'auto',
                 model_generation_params: dict = None,
                 api_rpc_port: int = DEFAULT_API_SERVER_RPC_PORT,
                 model_rpc_port: int = DEFAULT_MODEL0_SERVER_RPC_PORT):
        super().__init__(model_type, model_uri, model_device, model_generation_params, api_rpc_port, model_rpc_port)

    async def _rpc_step(self, rpc_obj: ImageToImageReq) -> any:
        if isinstance(rpc_obj, ImageToImageReq):
            result = await self._image_to_image(rpc_obj.image_data, rpc_obj.extra_params)
        else:
            assert False, 'Unsupported RPC object'
            # TODO Throw exception

        return result

    @abstractmethod
    async def _image_to_image(self, image_data: any, generation_params: Optional[dict] = None):
        pass
