#!/bin/bash

#sudo apt install -y nvidia-container-toolkit

docker build -f Dockerfile_api.amd64 ../../../ -t vfastml.apps.openai.api
docker build -f Dockerfile_model.amd64 ../../../ -t vfastml.apps.openai.model