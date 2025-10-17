from gsm_benchmarker.benchmark_config import BenchmarkConfig


class BaseModelWrapper:
    def __init__(self, model_name: str, config: BenchmarkConfig):
        self.config = config
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def ask(self, prompt: str) -> str:
        """Generate response from model"""

        raise NotImplementedError
