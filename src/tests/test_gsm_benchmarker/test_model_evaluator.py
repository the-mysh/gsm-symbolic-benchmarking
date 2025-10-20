import pytest

from gsm_benchmarker.model_evaluator import ModelEvaluator


@pytest.mark.parametrize(("resp", "value"), (
    ("The final answer is 42.", 42),
    ("The answer is 38", 38),
    ("Answer: 3.1", 3.1),
    ("answer:    22", 22),
    ("#### -1", -1),
    ("###### 3", 3),
    ("A: 31", 31),
    ("=2.5", 2.5),
    ("=      -23.8", -23.8),
    ("value=3.2\n", 3.2)
))
def test_extract_answer(resp, value):
    extracted_value = ModelEvaluator.extract_answer(resp)
    assert extracted_value == pytest.approx(value, abs=1e-5)
