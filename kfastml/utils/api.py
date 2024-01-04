import uuid
from datetime import datetime

from fastapi.responses import JSONResponse

from kfastml.engine.dispatch_engine import DispatchEngineTaskResult


def gen_request_id(api_name: str) -> str:
    return f'{api_name}_{uuid.uuid4().hex}'


def build_json_response(request_id: str, job_result: DispatchEngineTaskResult) -> JSONResponse:
    return JSONResponse({
        'created': f'{datetime.now()}',
        'id': request_id,

        'finished_reason': job_result.finished_reason,
        'result': job_result.result,
    })


def _is_package_available(package_name: str, min_version: str = None) -> bool:
    import importlib.metadata
    from packaging.version import Version
    try:
        package_version = importlib.metadata.version(package_name)
        if min_version is None:
            return True
        else:
            return Version(package_version) >= Version(min_version)

    except importlib.metadata.PackageNotFoundError:
        return False
