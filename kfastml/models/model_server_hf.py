import asyncio

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from kfastml.models.model_server import ModelServer
from kfastml.utils import DEFAULT_API_SERVER_RPC_PORT, DEFAULT_MODEL0_SERVER_RPC_PORT


class ModelServerHF(ModelServer):

    def __init__(self, model_type: str, model_uri: str, model_device: str = 'auto',
                 model_generation_params: dict = None, api_rpc_port: int = DEFAULT_API_SERVER_RPC_PORT,
                 model_rpc_port: int = DEFAULT_MODEL0_SERVER_RPC_PORT) -> None:
        super().__init__(model_type, model_uri, model_device, model_generation_params, api_rpc_port, model_rpc_port)

    async def _load_model(self):
        self.config = AutoConfig.from_pretrained(
            self.model_uri,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_uri,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_uri,
        )

        self._is_model_loaded = True
