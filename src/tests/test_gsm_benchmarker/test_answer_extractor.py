import logging
import pytest

from gsm_benchmarker.answer_extractor import AnswerExtractor, AnswerPattern


@pytest.mark.parametrize(("resp", "value", "pattern"), (
    ("The final answer is 42.", 42, AnswerPattern.GSM_SYMBOLIC),
    ("The answer is 38", 38, AnswerPattern.GSM_SYMBOLIC),
    ("#### -1", -1, AnswerPattern.GMS8K),
    ("###### 3", 3, AnswerPattern.GMS8K),
    ("=2.5", 2.5, AnswerPattern.EQUAL_SIGN),
    ("=      -23.8", -23.8, AnswerPattern.EQUAL_SIGN),
    ("value=3.2\n", 3.2, AnswerPattern.EQUAL_SIGN)
))
def test_extract_answer_from_pattern(resp, value, pattern, caplog):

    with caplog.at_level(logging.DEBUG):
        extracted_value, detected_pattern = AnswerExtractor.extract_answer(resp)

    assert extracted_value == pytest.approx(value, abs=1e-5)
    assert detected_pattern is pattern


@pytest.mark.parametrize(("resp", "value"), (
    ("Answer: 3.1", 3.1),
    ("answer:    22", 22),
    ("A: 31", 31),
    ("was 5, subtracted 1, left 4", 4)
))
def test_extract_answer_no_pattern(resp, value, caplog):

    with caplog.at_level(logging.DEBUG):
        extracted_value, detected_pattern = AnswerExtractor.extract_answer(resp)

    assert extracted_value == pytest.approx(value, abs=1e-5)
    assert detected_pattern is AnswerPattern.LAST_NUMBER

@pytest.mark.parametrize(("resp", "trimmed"), (
    ("some answer blah blah\n\nQ:", "some answer blah blah\n\n"),
    ("xyz!Q:ddd", "xyz!"),
    ("abc </s> saaaa", "abc "),
    ("response.<|endoftext|>", "response."),
    ("some text = 3 \n **`<", "some text = 3 \n "),
    ("blahblah<end_of_turn>\t\t\t", "blahblah"),
    ("The final answer is 42.[/INST]", "The final answer is 42."),
))
def test_trim_response(resp, trimmed):
    assert AnswerExtractor.trim_response(resp) == trimmed


@pytest.mark.parametrize("resp", (
    "The answer is 42!",
    "This is it<s>",
    "Or is it?**"
))
def test_trim_response_no_trim(resp):
    assert AnswerExtractor.trim_response(resp) == resp
