from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ModelHeaderPartition:
    vocab_embedding: nn.Embedding
    lm_head: nn.Linear


@dataclass
class ModelLayersPartition:
    layers: list[torch.nn.Module]


class MistralModelSingleGpuPartitioned:
    def __init__(self, header: ModelHeaderPartition, blocks: list[ModelLayersPartition]):
        self.header = header
        self.blocks = blocks

    @staticmethod
    #def from_hf(mistral: MistralForCausalLM):
    def from_hf(mistral_lm: any):
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
            lm_head=mistral_lm.lm_head)

        return MistralModelSingleGpuPartitioned(
            header=header_block,
            blocks=layers_blocks,
        )