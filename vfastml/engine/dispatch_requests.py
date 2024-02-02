from dataclasses import dataclass
from typing import Literal

from vfastml.errors import ErrorResult


class DispatchRequestResult:
    def __init__(self,
                 request_id: str,
                 result: dict | None,
                 error: ErrorResult | None = None,
                 ):
        assert type(request_id) is str
        assert result is None or error is None
        self.request_id = request_id
        self.result = result
        self.error = error

    def succeeded(self) -> bool:
        return self.error is None

    @staticmethod
    def from_exception(request_id: str, error: Exception):
        return DispatchRequestResult(request_id, None, ErrorResult.from_exception(error))


class BaseDispatchRequest:
    def __init__(self,
                 request_id: str,
                 ):
        if not isinstance(request_id, str):
            raise TypeError('request_id must be str')

        self.request_id = request_id


@dataclass
class TextGenerationMessage:
    role: Literal['system', 'user', 'assistant']
    content: str


@dataclass
class TextGenerationForward:
    # Generation config
    min_tokens: int | None = None
    max_tokens: int | None = None
    max_time: float | None = None
    num_generations: int = 1
    # Token generation config
    seed: int | None = 0
    repetition_penalty: float | None = 1.0
    temperature: float | None = 1.0
    top_p: float | None = 1.0
    logit_bias: dict[int, float] | None = None
    # TODO no_repeat_size, bad_words_ids, force_words_ids
    # Options
    output_scores: bool = False
    stream: bool = False
    use_cache: bool = True

    def __post_init__(self):
        if self.min_tokens and (self.min_tokens < 1 or self.min_tokens > 2048):
            raise ValueError('min_tokens must be between [1, 2048]')
        if self.max_tokens and (self.max_tokens < 1 or self.max_tokens > 2048):
            raise ValueError('max_tokens must be between [1, 2048]')
        if self.min_tokens and self.max_tokens and self.min_tokens > self.max_tokens:
            raise ValueError('min_tokens must be <= max_tokens')
        if self.num_generations and (self.num_generations < 1 or self.num_generations > 9):
            raise ValueError('num_generations must be between [1, 9]')
        # TODO Finish validation


class TextGenerationReq(BaseDispatchRequest):
    def __init__(self,
                 request_id: str,
                 messages: str | list[TextGenerationMessage],
                 forward_params: TextGenerationForward,
                 model_uri: str,
                 model_adapter_uri: str | None = None,
                 ):

        if not isinstance(messages, (str, list)):
            raise TypeError('messages must be [str | list[TextGenerationMessage]')
        if not isinstance(forward_params, TextGenerationForward):
            raise TypeError('forward_params must be TextGenerationForward')
        if not isinstance(model_uri, str):
            raise TypeError('model_uri must be str')
        if model_adapter_uri and not isinstance(model_adapter_uri, str):
            raise TypeError('model_adapter_uri must be str')

        if isinstance(messages, str):
            messages = [TextGenerationMessage(
                role = 'user',
                content = messages,
            )]

        super().__init__(request_id)
        self.messages = messages
        self.forward_params = forward_params
        self.model_uri = model_uri
        self.model_adapter_uri = model_adapter_uri


class ImageToImageReq(BaseDispatchRequest):
    def __init__(self,
                 request_id: str,
                 images: list[str | bytes],
                 forward_params: dict,
                 model_uri: str,
                 model_adapter_uri: str | None = None,
                 ):
        super().__init__(request_id)
        self.images = images
        self.forward_params = forward_params
        self.model_uri = model_uri
        self.model_adapter_uri = model_adapter_uri
