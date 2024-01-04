#!/bin/bash

pushd .
cd api
docker build -f Dockerfile.amd64 ../../../ -t kfastml.text_generation.api_server:v1
popd

pushd .
cd model
docker build -f Dockerfile.amd64 ../../../ -t kfastml.text_generation.model_server:v1
popd