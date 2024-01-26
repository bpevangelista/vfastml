from typing import Literal

import torch
import torch.nn as nn
from transformers import DynamicCache, LogitsProcessorList, StoppingCriteriaList, PreTrainedModel
from transformers.generation import MaxLengthCriteria, MaxTimeCriteria
# noinspection PyProtectedMember
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa, \
    _prepare_4d_causal_attention_mask


class ModelHeaderPartition(nn.Module):
    def __init__(self,
                 vocab_embedding: nn.Embedding,
                 lm_head: nn.Linear,
                 lm_norm: nn.Module,
                 ):
        super(ModelHeaderPartition, self).__init__()
        self.vocab_embedding = vocab_embedding
        self.lm_head = lm_head
        self.lm_norm = lm_norm


class ModelLayersPartition(nn.Module):
    def __init__(self, layers: list[nn.Module]):
        super(ModelLayersPartition, self).__init__()
        self.layers = nn.ModuleList(layers)

    def pin_memory(self):
        for layer in self.layers:
            layer.pin_memory()

class MistralModelSingleGpuPartitionedConfig:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2


class MistralModelSingleGpuPartitioned(nn.Module):
    def __init__(self,
                 header: ModelHeaderPartition,
                 blocks: list[ModelLayersPartition],
                 device: str,
                 attn_implementation: Literal['sdpa', 'flash_attention_2', 'manual'],
                 ):
        super(MistralModelSingleGpuPartitioned, self).__init__()
        self.header = header
        self.blocks = nn.ModuleList(blocks)
        self.device = device
        self.attn_implementation = attn_implementation

        # Pin all blocks and move all but first to CPU
        """
        for i in range(len(self.blocks)):
            self.blocks[i] = self.blocks[i].pin_memory()
            if i == 0:
                self.blocks[i] = self.blocks[i].to(self.device)
            else:
                self.blocks[i] = self.blocks[i].cpu()
        """


    def _skip_kv_cache_inputs(
            self,
            input_ids: torch.Tensor,
            attention_mask_2d: torch.Tensor,
            kv_cache: DynamicCache) -> tuple[torch.Tensor, torch.Tensor]:
        position_ids = attention_mask_2d.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask_2d == 0, 1)

        # First run only
        if kv_cache is not None:
            past_length = kv_cache.seen_tokens
            input_ids = input_ids[:, past_length:]
            position_ids = position_ids[:, past_length:]

        return input_ids, position_ids


    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                ):

        past_kv_cache = DynamicCache()
        next_decoder_cache = None
        logits_processor = LogitsProcessorList()

        max_new_tokens = 1024
        max_tokens = 2048
        stopping_criteria = StoppingCriteriaList([
            MaxLengthCriteria(min(len(input_ids) + max_new_tokens, max_tokens)),
            MaxTimeCriteria(60.0), # 1min timeout
        ])

        eos_token_id = MistralModelSingleGpuPartitionedConfig.eos_token_id
        eos_token_id_tensor = torch.tensor([eos_token_id]).to(self.device)

        batch_count, seq_length = input_ids.shape
        unfinished_batches = torch.ones(batch_count, dtype=torch.long, device=self.device)
        attention_mask_2d = attention_mask

        has_finished = False
        while not has_finished:
            unseen_input_ids, unseen_position_ids = self._skip_kv_cache_inputs(input_ids, attention_mask_2d, past_kv_cache)
            batch_count, seq_length = unseen_input_ids.shape
            hidden_states = self.header.vocab_embedding(unseen_input_ids)

            if self.attn_implementation == 'flash_attention_2':
                attention_mask = attention_mask_2d if (0 in attention_mask_2d) else None
            else:
                if self.attn_implementation == 'sdpa':
                    mask_prepare_func = _prepare_4d_causal_attention_mask_for_sdpa
                else:
                    mask_prepare_func =_prepare_4d_causal_attention_mask

                attention_mask = mask_prepare_func(
                    attention_mask=attention_mask_2d,
                    input_shape=(batch_count, seq_length),
                    inputs_embeds=hidden_states,
                    past_key_values_length=past_kv_cache.get_usable_length(seq_length),
                )

            block_count = len(self.blocks)
            gpu_blocks = [self.blocks[0]]
            for i in range(block_count):

                block = gpu_blocks.pop()
                next_block_idx = (i + 1) % block_count
                #gpu_blocks.append(self.blocks[next_block_idx].to(self.device))
                gpu_blocks.append(self.blocks[next_block_idx])

                for layer in block.layers:
                    outputs = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        position_ids=unseen_position_ids,
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

            # One new token per-batch
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)
            next_tokens = next_tokens * unfinished_batches + eos_token_id * (1 - unfinished_batches)
            # TODO Stream Out next_tokens.cpu()

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            next_attn_mask = attention_mask_2d.new_ones((attention_mask_2d.shape[0], 1))
            attention_mask_2d = torch.cat([attention_mask_2d, next_attn_mask], dim=-1)

            unfinished_batches = unfinished_batches.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            if unfinished_batches.max() == 0 or stopping_criteria(input_ids, scores=None):
                has_finished = True

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

        # noinspection PyProtectedMember
        attn_implementation = {
            'sdpa': 'sdpa',
            'flash_attention_2': 'flash_attention_2',
            'eager': 'manual',
        }.get(mistral_lm.model.config._attn_implementation, 'manual')

        return MistralModelSingleGpuPartitioned(
            header=header_block,
            blocks=layers_blocks,
            device=device,
            attn_implementation=attn_implementation,
        )

    @staticmethod
    def from_pretrained(path: str):
        pass