from typing import Optional


class DispatchEngineRequestResult:
    def __init__(self,
                 request_id: str,
                 finished_reason: str,
                 result: any,
                 ):
        assert type(request_id) is str
        self.request_id = request_id
        self.finished_reason = finished_reason
        self.result = result


class BaseDispatchEngineRequest:
    def __init__(self,
                 request_id: str,
                 ):
        assert type(request_id) is str
        self.request_id = request_id


class TextGenerationReq(BaseDispatchEngineRequest):
    def __init__(self,
                 request_id: str,
                 model_uri: str,
                 model_adapter_uri: Optional[str] = None,
                 prompt: str = None,
                 extra_params: Optional[dict] = None,
                 ):
        super().__init__(request_id)
        self.model_uri = model_uri
        self.model_adapter_uri = model_adapter_uri
        self.prompt = prompt
        self.extra_params = extra_params


class ImageToImageReq(BaseDispatchEngineRequest):
    def __init__(self,
                 request_id: str,
                 model_uri: str,
                 images: list[str | bytes],
                 extra_params: Optional[dict] = None,
                 ):
        super().__init__(request_id)
        self.model_uri = model_uri
        self.images = images
        self.extra_params = extra_params
