from typing import Literal
from pydantic import BaseModel


class ChatCompletionsMessage(BaseModel):
    role: Literal['user', 'assistant', 'system']
    content: str

class ChatCompletionsResponseFormat(BaseModel):
    type: Literal['json_object', 'text']

class ChatCompletionsRequest(BaseModel):
    messages: list[ChatCompletionsMessage] | str
    model: str
    frequency_penalty: float | None = 0.0
    logit_bias: dict[int, float] | None = None
    logprobs: bool | None = False
    top_logprobs: bool | None = False           # NotImplemented
    max_tokens: int | None = 2048
    n: int | None = 1
    presence_penalty: float | None = 0.0        # NotImplemented
    response_format: ChatCompletionsResponseFormat | None = None        # NotImplemented
    seed: int | None = None
    stop: str | list | None = None              # NotImplemented
    stream: bool | None = False
    temperature: int | None = 1.0
    top_p: int | None = 1.0
    tools: list | None = None                   # NotImplemented
    tool_choice: str | object | None = None     # NotImplemented
    user: str | None = None
