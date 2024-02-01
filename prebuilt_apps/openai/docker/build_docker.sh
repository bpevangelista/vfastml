#!/bin/bash

docker build -f Dockerfile_api.amd64 ../../ -t vfastml.apps.openai.api:v1
docker build -f Dockerfile_model.amd64 ../../ -t vfastml.apps.openai.model:v1