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
        assert type(request_id) is str
        self.request_id = request_id


class TextGenerationMessage:
    def __init__(self, role: Literal['system', 'user', 'assistant'], content: str):
        self.role = role
        self.content = content


class TextGenerationReq(BaseDispatchRequest):
    def __init__(self,
                 request_id: str,
                 model_uri: str,
                 model_adapter_uri: str | None = None,
                 messages: str | list[TextGenerationMessage] = None,
                 extra_params: dict | None = None,
                 ):
        super().__init__(request_id)
        self.model_uri = model_uri
        self.model_adapter_uri = model_adapter_uri
        self.messages = messages
        self.extra_params = extra_params


class ImageToImageReq(BaseDispatchRequest):
    def __init__(self,
                 request_id: str,
                 model_uri: str,
                 images: list[str | bytes],
                 extra_params: dict | None = None,
                 ):
        super().__init__(request_id)
        self.model_uri = model_uri
        self.images = images
        self.extra_params = extra_params
