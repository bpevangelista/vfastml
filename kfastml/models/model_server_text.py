from abc import ABC, abstractmethod

from kfastml.engine.dispatch_requests import TextGenerationReq, TextGenerationMessage
from kfastml.models.model_server import ModelServer


class TextGenerationModelServer(ModelServer, ABC):
    async def _rpc_step(self, rpc_obj: TextGenerationReq) -> dict:
        result = await self._generate_text(rpc_obj.request_id, rpc_obj.messages, **(rpc_obj.extra_params or {}))
        assert isinstance(result, dict), "_generate_text must return a dict"
        return result

    @abstractmethod
    async def _generate_text(self, request_id: str, prompt: str | list[TextGenerationMessage], **kwargs) -> dict:
        raise NotImplementedError
