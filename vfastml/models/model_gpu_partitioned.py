from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import DynamicCache, LogitsProcessorList, StoppingCriteriaList, PreTrainedModel
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask


@dataclass
class ModelHeaderPartition:
    vocab_embedding: nn.Embedding
    lm_head: nn.Linear
    lm_norm: nn.Linear


@dataclass
class ModelLayersPartition:
    layers: list[torch.nn.Module]


class MistralModelSingleGpuPartitionedConfig:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2


class MistralModelSingleGpuPartitioned(nn.Module):
    def __init__(self,
                 header: ModelHeaderPartition,
                 blocks: list[ModelLayersPartition],
                 device: str,
                 ):
        super().__init__()
        self.header = header
        self.blocks = blocks
        self.device = device

    def forward(self,
                input_ids: torch.Tensor,
                ):
        past_kv_cache = DynamicCache()
        logits_processor = LogitsProcessorList()
        stopping_criteria = StoppingCriteriaList()

        eos_token_id = MistralModelSingleGpuPartitionedConfig.eos_token_id
        eos_token_id_tensor = torch.tensor([eos_token_id]).to(self.device)

        batch_count, seq_length = input_ids.shape
        unfinished_batches = torch.ones(batch_count, dtype=torch.long, device=self.device)

        is_done = False
        while not is_done:
            batch_count, seq_length = input_ids.shape
            hidden_states = self.header.vocab_embedding(input_ids)

            past_kv_length = past_kv_cache.get_usable_length(seq_length)
            position_ids = torch.arange(past_kv_length,
                                        seq_length + past_kv_length,
                                        dtype=torch.long,
                                        device=self.device).unsqueeze(0)

            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask=None,
                input_shape=(batch_count, seq_length),
                inputs_embeds=hidden_states,
                past_key_values_length=past_kv_length,
            )

            for block in self.blocks: # per partition
                for layer in block.layers:
                    outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_kv_cache,
                        output_attentions=False,
                        use_cache=True,
                    )
                    hidden_states = outputs[0]
                    next_decoder_cache = outputs[1]

            past_kv_cache = next_decoder_cache

            hidden_states = self.header.lm_norm(hidden_states)
            all_token_logits = self.header.lm_head(hidden_states).float()

            next_token_logits = all_token_logits[:, -1, :]
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            next_tokens = next_tokens * unfinished_batches + eos_token_id * (1 - unfinished_batches)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)

            unfinished_batches = unfinished_batches.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            if unfinished_batches.max() == 0 or stopping_criteria(input_ids, ()):
                is_done = True

        return input_ids

    @staticmethod
    def from_hf(mistral_lm: PreTrainedModel, device: str):
        layers = list(mistral_lm.model.layers.children())
        layers_blocks: list[ModelLayersPartition] = []

        block_size = 4
        block_count = (len(layers) + block_size - 1) // block_size

        for i in range(block_count):
            block_start = i * block_size
            layers_cut = layers[block_start:block_start + block_size]
            layers_blocks.append(ModelLayersPartition(layers=layers_cut))

        header_block = ModelHeaderPartition(
            vocab_embedding=mistral_lm.model.embed_tokens,
            lm_head=mistral_lm.lm_head,
            lm_norm=mistral_lm.model.norm,
        )

        return MistralModelSingleGpuPartitioned(
            header=header_block,
            blocks=layers_blocks,
            device=device,
        )

    @staticmethod
    def from_pretrained(path: str):
        pass