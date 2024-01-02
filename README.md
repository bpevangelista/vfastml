# kFastML

Goal: One line command for YAML --> Inference Sever generation, docker image
build and k8s deployment.

### Architecture:

![Screenshot](docs/architecture.png)

### Why?:

- Easy serving and training of diverse ML pipelines
- Horizontal Scalability and Stability with k8s (deploy and forget)

### TODOs:

- Support continuous batching (per-iteration)
- For LLM, make sure we use flash-attention-v2 + (paged-attention |
  token-attention)

### Minors:

- Update excalidraw
- Wait model before REST API available