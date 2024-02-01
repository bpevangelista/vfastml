import argparse

from pydantic import BaseModel

from vfastml.engine.dispatch_requests import ImageToImageReq
# noinspection PyPep8Naming
from vfastml.entrypoints.api_server import FastMLServer as ml_server
from vfastml.utils import api


class ImageCleanupRequest(BaseModel):
    images_uri: list[str]
    model: str | None = None


# Upload Files Data with multipart/form-data & Forms
# https://stackoverflow.com/questions/65504438/how-to-add-both-file-and-json-body-in-a-fastapi-post-request
@ml_server.app.post(path='/v1/image/cleanup', status_code=200)
async def image_cleanup(request: ImageCleanupRequest):
    request_id = api.gen_request_id('cleanup')

    available_models = {
        'wdnet': {'model': 'wdnet_v0.1'},
    }
    selected_model = available_models.get(request.model, available_models['wdnet'])

    dispatch_task = ml_server.dispatch_engine.dispatch(
        ImageToImageReq(request_id=request_id,
                        model_uri=selected_model['model'],
                        images=request.images_uri))
    task_result = await dispatch_task.get_result()

    return api.build_json_response(request_id, task_result)


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglevel', type=str, default='debug')
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    return parser.parse_args()


if __name__ == '__main__':
    args = handle_args()

    ml_server.configure(log_level=args.log_level, dispatch_engine='async')
    ml_server.run(
        host=args.host,
        port=args.port,
    )
