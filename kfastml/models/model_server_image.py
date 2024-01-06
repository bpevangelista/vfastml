from abc import ABC, abstractmethod

from kfastml.engine.dispatch_requests import ImageToImageReq
from kfastml.models.model_server import ModelServer


class ImageToImageModelServer(ModelServer, ABC):
    async def _rpc_step(self, rpc_obj: ImageToImageReq) -> dict | None:
        if isinstance(rpc_obj, ImageToImageReq):
            result = await self._image_to_image(rpc_obj.request_id, rpc_obj.images,
                                                **(rpc_obj.extra_params or {}))
        else:
            # TODO Throw exception
            assert False, 'Unsupported RPC object'

        return result

    @abstractmethod
    async def _image_to_image(self,
                              request_id: str,
                              images: list[str | bytes],
                              **kwargs,
                              ) -> dict | None:
        assert False, 'Not Implemented'
