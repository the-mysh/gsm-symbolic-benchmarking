import pytest

from gsm_benchmarker.shot_manager import SingleShot, GSMShotManager


def test_shots_init(shot_manager: GSMShotManager):
    assert isinstance(shot_manager._shots, tuple)
    assert len(shot_manager._shots) == 8
    assert all(isinstance(s, SingleShot) for s in shot_manager._shots)


def test_mock_shots_init(mock_shot_manager: GSMShotManager):
    assert isinstance(mock_shot_manager._shots, tuple)
    assert len(mock_shot_manager._shots) == 3
    assert all(isinstance(s, SingleShot) for s in mock_shot_manager._shots)


@pytest.mark.parametrize(("fmt", "n_shots", "separator", "res"), (
    (
        "Q: {question}\nA: Let's think step by step. {solution} The final answer is {result}.",
        2,
        "\n\n",
        """Q: Q1?
A: Let's think step by step. A1. The final answer is 11.

Q: Q2?
A: Let's think step by step. A2. The final answer is 25."""
    ),
    (
        "Question: {question} Answer: {solution} {result}",
        3,
        "\n",
        "Question: Q1? Answer: A1. 11\nQuestion: Q2? Answer: A2. 25\nQuestion: Q3? Answer: A3. 39"
    )
))
def test_shots_compile(mock_shot_manager: GSMShotManager, fmt, n_shots, separator, res):
    assert mock_shot_manager.compile(fmt, n_shots=n_shots, separator=separator) == res
