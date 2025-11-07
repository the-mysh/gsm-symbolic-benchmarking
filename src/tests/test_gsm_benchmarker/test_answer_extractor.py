import logging
import pytest

from gsm_benchmarker.answer_extractor import AnswerExtractor


@pytest.mark.parametrize(("resp", "value", "pattern"), (
    ("The final answer is 42.", 42, "GSM-Symbolic format"),
    ("The answer is 38", 38, "GSM-Symbolic format"),
    ("Answer: 3.1", 3.1, "'Answer:' format"),
    ("answer:    22", 22, "'Answer:' format"),
    ("#### -1", -1, "GSM8K standard format"),
    ("###### 3", 3, "GSM8K standard format"),
    ("=2.5", 2.5, "'= <number> format'"),
    ("=      -23.8", -23.8, "'= <number> format'"),
    ("value=3.2\n", 3.2, "'= <number> format'")
))
def test_extract_answer_from_pattern(resp, value, pattern, caplog):

    with caplog.at_level(logging.DEBUG):
        extracted_value = AnswerExtractor.extract_answer(resp)

    assert extracted_value == pytest.approx(value, abs=1e-5)
    assert "No predefined answer pattern" not in caplog.text
    assert pattern in caplog.text


@pytest.mark.parametrize(("resp", "value"), (
    ("A: 31", 31),
    ("was 5, subtracted 1, left 4", 4)
))
def test_extract_answer_no_pattern(resp, value, caplog):

    with caplog.at_level(logging.DEBUG):
        extracted_value = AnswerExtractor.extract_answer(resp)

    assert extracted_value == pytest.approx(value, abs=1e-5)
    assert "No predefined answer pattern" in caplog.text
    assert "Extracting answer as the last number" in caplog.text


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
