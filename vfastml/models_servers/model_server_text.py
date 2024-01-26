from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

from vfastml.engine.dispatch_requests import TextGenerationReq, TextGenerationMessage, TextGenerationForward
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
    async def _rpc_step(self, rpc_obj: TextGenerationReq) -> dict:
        result = await self._generate_text(rpc_obj.request_id, rpc_obj.messages, rpc_obj.forward_params)
        # TODO Change to a less openai like struct
        assert isinstance(result, ChatCompletionsResult), "_generate_text must return a ChatCompletionsResult"
        return result.model_dump()

    @abstractmethod
    async def _generate_text(self,
                             request_id: str,
                             prompt: str | list[TextGenerationMessage],
                             forward_kwargs: TextGenerationForward) -> ChatCompletionsResult:
        raise NotImplementedError
