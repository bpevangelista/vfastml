import uuid
from datetime import datetime

from fastapi.responses import JSONResponse

from vfastml.engine.dispatch_requests import DispatchRequestResult


def gen_request_id(api_name: str) -> str:
    return f'{api_name}_{uuid.uuid4().hex}'


def build_json_response(request_id: str, task_result: DispatchRequestResult) -> JSONResponse:
    if task_result.succeeded():
        status_code = 200
        response_json = {
            'id': request_id,
            'created': f'{datetime.now()}',
        }

        if isinstance(task_result.result, dict):
            response_json.update(task_result.result)
        else:
            response_json['result'] = task_result.result

    else:
        status_code = task_result.error.status_code
        response_json = task_result.error.error

    return JSONResponse(status_code=status_code, content=response_json)
