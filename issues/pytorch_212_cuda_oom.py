import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda:0")

tokenizer = AutoTokenizer.from_pretrained(
    # 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistralai/Mistral-7B-v0.1',
    trust_remote_code=True,
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    # 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistralai/Mistral-7B-v0.1',
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

generated_tokens = tokenizer("Tell me a joke", return_tensors="pt")

model_params = {
    'do_sample': True,
    'max_new_tokens': 64,
    'input_ids': generated_tokens.input_ids,
    'attention_mask': generated_tokens.attention_mask,
}

output = model.generate(**model_params)
print(tokenizer.batch_decode(output)[0])
