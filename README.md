# VFastML 

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
from vfastml.models_servers.model_server_text_hf import TextGenerationModelServerHF

model_server = TextGenerationModelServerHF(
    model_type='text_generation',
    model_uri='mistralai/Mistral-7B-v0.1',
    model_device='cuda:0',
    model_forward_kwargs={
        'top_p': 0.9,
    },
    log_level='debug',
)
model_server.run_forever()
```
``` python
# API Server
from vfastml.engine.dispatch_requests import TextGenerationReq
from vfastml.entrypoints.api_server import FastMLServer as ml_server

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

### Next Steps?

No Code! Just YAML &rarr; Inference server docker image and kubernets deployments.

``` YAML
apis:
  - path: /v1/chat/completions
    inputs:
      - messages: list[dict] | str
      - model: str
        optional: true
    outputs:
      - choices: list[dict]
servers:
  - model:
      type: text_generation
      uri: mistralai/Mistral-7B-v0.1
      device: cuda:0
      generation_params:
        top_p: 0.9
    resources:
      cpus: 2
      memory: 16GB
      gpus: 1
      gpu_memory: 24GB
      gpu_cuda_capability: 9.0
    rpc_port: 6500
```

### TODOs:

X-Large:
- Support continuous batching (per-iteration)

Core:
- Wait model_server before REST API available
- Support api_server->models router (for multi-model support)

Models
- Support heterogeneous sampling on same TextGen batch (beam and sample)
- Refactor wdnet and move to models
- Add whisper model support

Performance:
- Profile and calculate optimal batch size on start
- Implement benchmark for TGI and HF
  - Actually count generated tokens (don't trust it respect forward_params)

Stability & QoL:
- Add and improve input validation
- Refactor TextGeneration classes (internals and openai are a bit mixed)
- Expose classes through main package (avoid random imports)


### Frequent Issues:

#### Docker GPU build / run Issues?
- sudo apt-get install -y nvidia-container-toolkit
- docker run --gpus all -e HF_ACCESS_TOKEN=TOKEN vfastml.apps.openai.model:v1


#### Torch profiler not working on WSL2?

- nVidia Control Panel &rarr; Developer &rarr; Manage GPU Performance Counters &rarr;
Allow Access To All users
- Windows Settings &rarr; System &rarr; For Developers &rarr; Developer Mode ON

#### Incorrect CUDA version? (We are on cu118)
- pip install --upgrade torch==2.0.1+cu118 -f https://download.pytorch.org/whl/torch_stable.html

#### How create ssh key pair?

- ssh-keygen -t ed25519 -C your@email.com -f any_ed25519
  - eval $(ssh-agent -s) ssh-add ~/.ssh/any_ed25519
  - add to ~/.bashrc or ~/.zshrc: "IdentityFile ~/.ssh/any_ed25519"