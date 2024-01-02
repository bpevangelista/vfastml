#!/bin/bash

pushd .
cd api
docker build -f Dockerfile.amd64 ../../../ -t kfastml.text_generation.api_server:amd64:v1
popd

pushd .
cd model
docker -f Dockerfile.amd64 ../../../ -t kfastml.text_generation.model_server:amd64:v1
popd