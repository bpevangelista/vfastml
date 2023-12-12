import uuid
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI()
inference_engine = None

@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)

@app.post(path='/images/cleanup', status_code=200)
async def cleanup_images(request: Request):
    assert inference_engine is not None
    request_id = uuid.uuid4()
    request_dict = await request.json()
    request_params = {
        'path': '/images/cleanup',
        'images': request_dict.pop('image_urls')
    }

    job_result = inference_engine.enqueueJob(request_id, request_params)

    job_result = await job_result
    assert job_result is list

    result = {
        'ímages': job_result
    }
    return JSONResponse(result)
