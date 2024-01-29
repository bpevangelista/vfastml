import traceback
from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from vfastml import log
from vfastml.engine.dispatch_requests import TextGenerationReq, DispatchRequestResult
from vfastml.errors import InternalServerError
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
    async def _rpc_step(self):
        requests_batch: list[TextGenerationReq] = await self._get_request_batch(max_size=32)
        if requests_batch:
            # noinspection PyBroadException
            try:
                await self._generate_text(requests_batch)
            except Exception:
                log.error(traceback.format_exc())
                for request in requests_batch:
                    result = DispatchRequestResult.from_exception(request.request_id, InternalServerError())
                    await self._results_queue.put(result)


    @abstractmethod
    async def _generate_text(self, requests: list[TextGenerationReq]):
        raise NotImplementedError
