#!/bin/bash

docker build -f Dockerfile_api.amd64 ../../ -t kfastml.apps.openai.api:v1
docker build -f Dockerfile_model.amd64 ../../ -t kfastml.apps.openai.model:v1