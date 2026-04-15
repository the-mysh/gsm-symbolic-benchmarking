# written with the help of Gemini

import logging
import pytest

from gsm_benchmarker.benchmark.answer_extractor import AnswerExtractor, AnswerPattern


@pytest.mark.parametrize(("resp", "value", "pattern"), (
    ("The final answer is 42.", 42, AnswerPattern.GSM_SYMBOLIC),
    ("The answer is 38", 38, AnswerPattern.GSM_SYMBOLIC),
    ("#### -1", -1, AnswerPattern.GMS8K),
    ("###### 3", 3, AnswerPattern.GMS8K),
    ("=2.5", 2.5, AnswerPattern.EQUAL_SIGN),
    ("=      -23.8", -23.8, AnswerPattern.EQUAL_SIGN),
    ("value=3.2\n", 3.2, AnswerPattern.EQUAL_SIGN)
))
def test_extract_answer_textual_from_pattern(resp, value, pattern, caplog):

    with caplog.at_level(logging.DEBUG):
        extracted_value, detected_pattern = AnswerExtractor.extract_answer_textual(resp)

    assert extracted_value == pytest.approx(value, abs=1e-5)
    assert detected_pattern is pattern


@pytest.mark.parametrize(("resp", "value"), (
    ("Answer: 3.1", 3.1),
    ("answer:    22", 22),
    ("A: 31", 31),
    ("was 5, subtracted 1, left 4", 4)
))
def test_extract_answer_textual_no_pattern(resp, value, caplog):

    with caplog.at_level(logging.DEBUG):
        extracted_value, detected_pattern = AnswerExtractor.extract_answer_textual(resp)

    assert extracted_value == pytest.approx(value, abs=1e-5)
    assert detected_pattern is AnswerPattern.LAST_NUMBER

@pytest.mark.parametrize(("resp", "trimmed"), (
    ("some answer blah blah\n\nQ:", "some answer blah blah\n\n"),
    ("xyz!Q:ddd", "xyz!"),
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


def test_code_clean_explicit_definition():
    """Test a perfect, standard output with explicit def."""
    raw = """
    # given
    x = 1
    return x
"""
    result, _ = AnswerExtractor.extract_function_definition(raw)
    assert result  # not empty
    assert "def solution():" in result
    assert "return x" in result


def test_implicit_definition_prepend():
    """Test when model outputs only the body (common in few-shot)."""
    # The model skips 'def solution():' and starts identifying variables
    raw = """
    # given
    a = 10
    # calculation
    b = a * 2
    return b
"""

    result, _ = AnswerExtractor.extract_function_definition(raw)
    assert result  # not empty
    assert result.startswith("def solution():")  # Should have prepended header
    assert "b = a * 2" in result


def test_internal_newlines():
    """Test that empty lines inside the function don't break extraction."""
    raw = """def solution(): 
    x = 1
    # calculation
    y = x + 1

    return y"""

    result, _ = AnswerExtractor.extract_function_definition(raw)
    assert result  # not empty
    assert "return y" in result


@pytest.mark.parametrize("indent", ["  ", "\t", "    "])
def test_variable_indentation(indent): 
    """Test that 2 spaces, 4 spaces, and tabs are all accepted."""
    raw = f"""\ndef solution():\n{indent}# given\n{indent}x = 10\n{indent}return x """
    result, _ = AnswerExtractor.extract_function_definition(raw)
    assert result  # not empty


def test_garbage_input(): 
    """Test completely irrelevant input.""" 
    raw = "I cannot answer this question as it violates safety policies." 
    assert AnswerExtractor.extract_function_definition(raw)[0] == ""


def test_nested_markdown_mess(): 
    """Test messy markdown formatting often seen in smaller models."""
    raw = """ Sure:

    # given
    a = 5
    return a
""" 
    # This checks implicit def + markdown wrappers 
    result, _ = AnswerExtractor.extract_function_definition(raw)
    assert result  # not empty
    assert "def solution():" in result 
    assert "return a" in result
