from abc import ABC, abstractmethod

from kfastml.engine.dispatch_requests import ImageToImageReq
from kfastml.models.model_server import ModelServer


class ImageToImageModelServer(ModelServer, ABC):
    async def _rpc_step(self, rpc_obj: ImageToImageReq) -> dict:
        result = await self._image_to_image(rpc_obj.request_id, rpc_obj.images, **(rpc_obj.extra_params or {}))
        assert isinstance(result, dict), "_image_to_image must return a dict"
        return result

    @abstractmethod
    async def _image_to_image(self, request_id: str, images: list[str | bytes], **kwargs) -> dict:
        raise NotImplementedError
