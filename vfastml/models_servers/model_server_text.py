from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from vfastml.engine.dispatch_requests import TextGenerationReq, TextGenerationMessage, TextGenerationForward, \
    BaseDispatchRequest
from vfastml.models_servers.model_server import ModelServer


# TODO Move all below to prebuilt_apps and define simpler non-openai interfaces without extending BaseModel
class ChatCompletionsResultChoiceMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsResultLogProbsItem(BaseModel):
    token: str
    logprob: float
    bytes: list[int] | None

class ChatCompletionsResultLogProbs(ChatCompletionsResultLogProbsItem):
    top_logprobs: list[ChatCompletionsResultLogProbsItem]

class ChatCompletionsResultLogProbsContent(BaseModel):
    content: list[ChatCompletionsResultLogProbs] | None

class ChatCompletionsResultChoice(BaseModel):
    finish_reason: Literal['stop', 'length', 'content_filter']
    index: int
    message: ChatCompletionsResultChoiceMessage
    logprobs: ChatCompletionsResultLogProbsContent | None = None

class ChatCompletionsResultUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class ChatCompletionsResult(BaseModel):
    choices: list[ChatCompletionsResultChoice]
    model: str
    system_fingerprint: str
    object: str
    usage: ChatCompletionsResultUsage


class TextGenerationModelServer(ModelServer, ABC):
    def _requests_queue_copy(self) -> list[TextGenerationReq] | None:
        requests = self._requests_queue
        return requests.copy() if len(requests) > 0 and isinstance(requests[0], TextGenerationReq) else None


    async def _get_request_batch(self, min_size: int = 1, max_size: int = -1) -> list[TextGenerationReq] | None:
        result_batch: list[TextGenerationReq] | None = None

        await self._has_requests.wait()
        requests_count = len(self._requests_queue)

        if requests_count >= min_size:
            if max_size == -1 or requests_count <= max_size:
                result_batch = self._requests_queue_copy()
                self._requests_queue.clear()
                self._has_requests.clear()
            else:
                result_batch = self._requests_queue[:max_size]
                del self._requests_queue[:max_size]

            self._requests_queue_not_full.set()
        return result_batch


    async def _rpc_step(self):
        requests_batch: list[TextGenerationReq] = await self._get_request_batch(min_size=1, max_size=32)
        if requests_batch:
            await self._generate_text(requests_batch)
            self._has_results.set()


    @abstractmethod
    async def _generate_text(self, requests: list[TextGenerationReq]):
        raise NotImplementedError
