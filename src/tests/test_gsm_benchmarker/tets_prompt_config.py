import pytest

from gsm_benchmarker.shot_manager import GSMShotManager
from gsm_benchmarker.prompt_config import PromptConfig


@pytest.fixture
def default_pc():
    return PromptConfig(n_shots=3)


def test_default_pc(default_pc: PromptConfig, mock_shot_manager: GSMShotManager):
    prompt = default_pc(question="What about this one?", shots=mock_shot_manager)

    assert prompt == """As an expert problem solver, solve step by step the following mathematical questions.

Q: Q1?
A: Let's think step by step. A1. The final answer is 11.

Q: Q2?
A: Let's think step by step. A2. The final answer is 25.

Q: Q3?
A: Let's think step by step. A3. The final answer is 39.

Q: What about this one?
A: Let's think step by step."""

