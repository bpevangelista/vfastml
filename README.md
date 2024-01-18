# kFastML

Goal: One line command for YAML --> Inference Sever generation, docker image
build and k8s deployment.

### Architecture:

![Screenshot](docs/architecture.png)

### Running Prebuilt Apps
``` bash
pip install -r requirements.txt
python -m prebuilt_apps.openai.api_server   # run on bash 1
python -m prebuilt_apps.openai.model_server # run on bash 2
```

### Writing Your Own Apps
``` python
# Model Server
from kfastml.models_servers.model_server_text_hf import TextGenerationModelServerHF

model_server = TextGenerationModelServerHF(
    model_type='text_generation',
    model_uri='mistralai/Mistral-7B-v0.1',
    model_device='cuda:0',
    model_forward_kwargs={
        'top_p': 0.9,
    },
    log_level='debug',
)
model_server.run()
```
``` python
# API Server
from kfastml.engine.dispatch_requests import TextGenerationReq
from kfastml.entrypoints.api_server import FastMLServer as ml_server

@ml_server.app.post(path='/v1/chat/completions')
async def chat_completions(request: ChatCompletionsRequest):
    request_id = api_utils.gen_request_id('completions')
    
    selected_model = {
        'mistral': 'mistralai/Mistral-7B-v0.1',
        'phi2': 'microsoft/phi-2',
    }.get(request.model, 'mistralai/Mistral-7B-v0.1')

    dispatch_task = ml_server.dispatch_engine.dispatch(
        TextGenerationReq(
            request_id=request_id,
            model_uri=selected_model,
            model_adapter_uri=None,
            messages=request.messages))
    
    task_result = await dispatch_task.get_result()
    return api_utils.build_json_response(request_id, task_result)
```

### Why?

- Easy serving and training of diverse ML pipelines
- Horizontal Scalability and Stability with k8s (deploy and forget)

### TODOs:

- Support continuous batching (per-iteration)
- For LLM, huggingface uses flash-attn v2

### Minors:

- Wait model_server before REST API available
- Support api_server to N-models router (for multi-engine support)