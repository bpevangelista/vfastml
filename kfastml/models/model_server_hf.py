from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
import asyncio

from kfastml.models.model_server import ModelServer


class ModelServerHF(ModelServer):
    def __init__(self, server_port: int, server_type: str, model_uri: str, model_device: str, model_params: dict):
        super().__init__(server_port, server_type, model_uri, model_device, model_params)
        self.config = None
        self.tokenizer = None
        self.model = None

    async def _load_model(self):
        await asyncio.sleep(2)

        self.config = AutoConfig.from_pretrained(
            self.model_uri,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_uri
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_uri,
            config=self.config,
        )

        self._is_model_loaded = True
