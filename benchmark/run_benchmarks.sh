#!/bin/bash

sudo apt-get install huggingface-cli
huggingface-cli login

pip install -r requirements.txt
# Override vllm pydantic
pip install pydantic==2.3.0

python generate_dataset.py
python benchmark.py --max-prompts 200 vllm
python benchmark.py --max-prompts 200 vfastml
#python benchmark.py --max-prompts 200 tgi
#python benchmark.py --max-prompts 200 hf