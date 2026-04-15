import pytest

from gsm_benchmarker.input_data_management.shot_manager import GSMShotManager
from gsm_benchmarker.input_data_management.prompt_config import PromptConfig


@pytest.fixture
def default_pc(mock_shot_manager):
    pc = PromptConfig.default(n_shots=3)
    pc.shots = mock_shot_manager
    return pc


def test_default_pc(default_pc: PromptConfig, mock_shot_manager: GSMShotManager):
    prompt = default_pc(question="What about this one?")

    assert prompt == """As an expert problem solver, solve step by step the following mathematical questions.

Q: Q1?
A: Let's think step by step. A1. The final answer is 11.

Q: Q2?
A: Let's think step by step. A2. The final answer is 25.

Q: Q3?
A: Let's think step by step. A3. The final answer is 39.

Q: What about this one?
A: Let's think step by step. """


def test_pc_with_target_intro(mock_shot_manager):
    pc = PromptConfig.default(n_shots=2, target_intro="Now please solve this problem:")
    pc.shots = mock_shot_manager
    prompt = pc(question="What about this one?")

    assert prompt == """As an expert problem solver, solve step by step the following mathematical questions.

Q: Q1?
A: Let's think step by step. A1. The final answer is 11.

Q: Q2?
A: Let's think step by step. A2. The final answer is 25.

Now please solve this problem:

Q: What about this one?
A: Let's think step by step. """


def test_alternative_pc(mock_shot_manager):
    pc = PromptConfig(
        n_shots=2,
        target_intro="Now please solve this problem:",
        intro="Here are some example problems with answers:",
        question_format="Question: {question}",
        answer_format=" Answer: {solution} Final answer: {result}.",
        separator="\n"
    )
    pc.shots = mock_shot_manager
    prompt = pc(question="QQ?")

    assert prompt == """Here are some example problems with answers:
Question: Q1? Answer: A1. Final answer: 11.
Question: Q2? Answer: A2. Final answer: 25.
Now please solve this problem:
Question: QQ?"""


def test_pc_from_preset(mock_shot_manager):
    pc = PromptConfig.from_preset('separated-target', n_shots=1, shot_intro="Example {sid}:\n")
    pc.shots = mock_shot_manager
    prompt = pc(question="QQ?")

    assert prompt == """Here are some example problems with answers:

Example 1:
Q: Q1?
A: A1. The final answer is: 11.

Based on the examples above, please solve the problem below similarly.

Q: QQ?
A:"""
