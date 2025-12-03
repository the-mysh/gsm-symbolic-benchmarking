import re
import logging
from enum import Enum, auto
import traceback


logger = logging.getLogger(__name__)


class AnswerPattern(Enum):
    GMS8K = auto()
    GSM_SYMBOLIC = auto()
    EQUAL_SIGN = auto()
    LAST_NUMBER = auto()


class AnswerExtractionError(RuntimeError):
    pass


class AnswerExtractor:
    _number_pattern = r'\s*(?P<number>(-?\d+(?:\.\d+)?))'

    ANSWER_PATTERNS = {
        AnswerPattern.GMS8K: re.compile(r'####' + _number_pattern),
        AnswerPattern.GSM_SYMBOLIC: re.compile(r'[Tt]he (?:final )?answer is:?\s*\$?' + _number_pattern),
        AnswerPattern.EQUAL_SIGN: re.compile(r'=' + _number_pattern)
    }

    FUNCTION_PATTERN = re.compile(r"def (?P<func_name>\w+)\(\):\n(( {4}.+)?\n*)+")

    STOP_TOKENS = (
        # from paper
        "Q:",  # model moves on to generating a next question
        "Question:",  # same as above
        "</s>",
        "<|endoftext|>",

        # other, suggested by Gemini
        "**`<",  # OpenAI / Mistral / LLama3
        "<end_of_turn>",  # Gemma
        "[/INST]"  # Mistral Dialogic / Tool Use
    )

    def __init__(self, code: bool = False):
        self._extraction_method = self.extract_answer_code if code else self.extract_answer_textual

    def extract_answer(self, text: str) -> tuple[float | None, AnswerPattern | None]:
        """
        Extract numerical answer from text.
        Looks for patterns like "#### NUMBER" or "The (final) answer is NUMBER"
        """

        try:
            res = self._extraction_method(text)
        except AnswerExtractionError:
            logger.warning(f"Could not extract answer from model response:\n{text}")
            logger.warning(f"Extraction error stack:\n{traceback.format_exc()}")
            return None, None
        else:
            return res

    @classmethod
    def extract_answer_textual(cls, text: str) -> tuple[float | int, AnswerPattern]:
        text = cls.trim_response(text)

        for pattern_enum, pattern in cls.ANSWER_PATTERNS.items():
            match = pattern.search(text)
            if match:
                return float(match.group('number')), pattern_enum

        # Last resort if none of the patterns work: find last number in text
        numbers = re.findall(cls._number_pattern, text)
        if numbers:
            return float(numbers[-1][0]), AnswerPattern.LAST_NUMBER

        raise AnswerExtractionError(f"Could not locate numerical answer")

    @classmethod
    def extract_answer_code(cls, text: str) -> tuple[float | int, None]:
        text = cls.trim_response(text)

        match = cls.FUNCTION_PATTERN.search(text)
        if not match:
            raise AnswerExtractionError("Failed to find valid function definition in text")

        func_def = match.group()
        logger.debug(f"Matched function definition:\n{func_def}")
        loc = {}
        try:
            exec(f"{match.group()}\nret = {match.group('func_name')}()", locals(), loc)
        except SyntaxError:
            raise AnswerExtractionError(f"Extracted function definition has invalid syntax")
        except Exception as exc:
            raise AnswerExtractionError(f"Failed to obtain numerical answer by running extracted function: {exc}")

        res = loc['ret']
        if not isinstance(res, (int, float)):
            raise AnswerExtractionError(f"The result returned by the extracted function "
                                        f"({res}, type: {type(res).__name__}) is not a number")
        return res, None

    @classmethod
    def trim_response(cls, text: str) -> str:
        """'Trim' model response to the appearance of an end-of-response token - if any."""

        for stop_token in cls.STOP_TOKENS:
            idx = text.find(stop_token)
            if idx >= 0:  # -1 if not found
                return text[:idx]  # don't look for other stop tokens

        return text  # return original text if no stop tokens found
