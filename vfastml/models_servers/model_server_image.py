import traceback
from abc import ABC, abstractmethod

from vfastml import log
from vfastml.engine.dispatch_requests import ImageToImageReq, DispatchRequestResult
from vfastml.errors import InternalServerError
from vfastml.models_servers.model_server import ModelServer


class ImageToImageModelServer(ModelServer, ABC):
    async def _rpc_step(self):
        requests_batch: list[ImageToImageReq] = await self._get_request_batch(max_size=1)
        if requests_batch:
            # noinspection PyBroadException
            try:
                await self._image_to_image(requests_batch)
            except Exception:
                log.error(traceback.format_exc())
                for request in requests_batch:
                    result = DispatchRequestResult.from_exception(request.request_id, InternalServerError())
                    await self._results_queue.put(result)


    @abstractmethod
    async def _image_to_image(self, requests: list[ImageToImageReq]):
        raise NotImplementedError
