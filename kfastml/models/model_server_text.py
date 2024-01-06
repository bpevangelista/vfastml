from abc import ABC, abstractmethod

from kfastml.engine.dispatch_requests import TextGenerationReq
from kfastml.models.model_server import ModelServer


class TextGenerationModelServer(ModelServer, ABC):
    async def _rpc_step(self, rpc_obj: TextGenerationReq) -> dict | None:
        if isinstance(rpc_obj, TextGenerationReq):
            result = await self._generate_text(rpc_obj.prompt, **(rpc_obj.extra_params or {}))
        else:
            # TODO Throw exception
            assert False, 'Unsupported RPC object'

        return result

    @abstractmethod
    async def _generate_text(self, prompt: str, **kwargs) -> dict | None:
        assert False, 'Not Implemented'
