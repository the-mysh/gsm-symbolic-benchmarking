import re
import logging
from enum import Enum, auto


logger = logging.getLogger(__name__)


class AnswerPattern(Enum):
    GMS8K = auto()
    GSM_SYMBOLIC = auto()
    EQUAL_SIGN = auto()
    LAST_NUMBER = auto()
    CODE = auto()


class AnswerExtractionError(RuntimeError):
    pass


class ErrorType(Enum):
    NO_NUMBER = auto()  # for textual answers - when a number could not be extracted
    NO_FUNCTION = auto()  # failed to extract function definition
    SYNTAX = auto()  # function definition extracted, but has invalid syntax
    FORBIDDEN_STRING = auto()  # one of the potentially dangerous strings (e.g. 'eval') found in function
    NOT_A_NUMBER = auto()  # for code answers - when the return value of a function is not a number
    UNCLASSIFIED = auto()  # all others


# builtins and imports we can let the generated code use
SAFE_BUILTINS = {
    'int': int, 'float': float, 'str': str, 'bool': bool, 'complex': complex,
    'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
    'len': len, 'range': range, 'enumerate': enumerate, 'zip': zip,
    'all': all, 'any': any, 'reversed': reversed, 'sorted': sorted, 'filter': filter, 'map': map, 'slice': slice,
    'round': round, 'min': min, 'max': max, 'abs': abs, 'sum': sum, 'pow': pow, 'divmod': divmod,
    'print': print
}

SAFE_IMPORTS = {
    'numbers': __import__('numbers'),
    'math': __import__('math'),
    'cmath': __import__('cmath'),
    'decimal': __import__('decimal'),
    'fractions': __import__('fractions'),
    'random': __import__('random'),
    'typing': __import__('typing'),
    'array': __import__('array')
}


class AnswerExtractor:
    _number_pattern = r'\s*(?P<number>(-?\d+(?:\.\d+)?))'

    ANSWER_PATTERNS = {
        AnswerPattern.GMS8K: re.compile(r'####' + _number_pattern),
        AnswerPattern.GSM_SYMBOLIC: re.compile(r'[Tt]he (?:final )?answer is:?\s*\$?' + _number_pattern),
        AnswerPattern.EQUAL_SIGN: re.compile(r'=' + _number_pattern)
    }

    FUNCTION_PATTERN = re.compile(r"^def (?P<func_name>\w+)\(\):\n(( {4}.*)?\n*)+", flags=re.MULTILINE)
    FORBIDDEN_ITEMS = [
        re.compile(r"open\(.*\)"),
        re.compile(r"eval\(.*\)"),
        re.compile(r"exec\(.*\)"),
        re.compile(r"__import__\(.*\)"),
        re.compile(r"[gs]etattr\(.*\)")
    ]

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

    def extract_answer(self, text: str) -> tuple[float | int | None, AnswerPattern | ErrorType | None]:
        """
        Extract numerical answer from text.
        Looks for patterns like "#### NUMBER" or "The (final) answer is NUMBER"
        """

        res, answer_pattern_or_error_type = self._extraction_method(text)
        if res is None:
            logger.warning(f"-> Could not extract answer from model response:\n{text}")
        return res, answer_pattern_or_error_type

    @classmethod
    def extract_answer_textual(cls, text: str) -> tuple[float | int | None, AnswerPattern | ErrorType]:
        text = cls.trim_response(text)

        for pattern_enum, pattern in cls.ANSWER_PATTERNS.items():
            match = pattern.search(text)
            if match:
                return float(match.group('number')), pattern_enum

        # Last resort if none of the patterns work: find last number in text
        numbers = re.findall(cls._number_pattern, text)
        if numbers:
            return float(numbers[-1][0]), AnswerPattern.LAST_NUMBER

        logger.warning(f"Could not locate numerical answer")
        return None, ErrorType.NO_NUMBER

    @classmethod
    def check_extracted_func(cls, func_def: str):
        for s in cls.FORBIDDEN_ITEMS:
            if (m := s.search(func_def)) is not None:
                logger.warning(f"Potentially dangerous string ('{m.group()}') found in the extracted function")
                return True
        return False

    @classmethod
    def extract_function_definition(cls, text: str) -> tuple[str, str]:
        text = cls.trim_response(text)

        match = cls.FUNCTION_PATTERN.search(text)
        if not match:
            return "", ""

        return match.group(), match.group('func_name')

    @classmethod
    def extract_answer_code(cls, text: str) -> tuple[float | int | None, AnswerPattern | ErrorType]:
        func_def, func_name = cls.extract_function_definition(text)

        if not func_def:
            logger.warning("Failed to find valid function definition in text")
            return None, ErrorType.UNCLASSIFIED

        if cls.check_extracted_func(func_def):
            return None, ErrorType.FORBIDDEN_STRING

        scope = {'__builtins__': SAFE_BUILTINS.copy(), **SAFE_IMPORTS}
        loc = {}
        code = f"{func_def}\nret = {func_name}()"
        try:
            exec(code, scope, loc)
        except SyntaxError:
            logger.warning(f"Extracted function definition has invalid syntax")
            return None, ErrorType.SYNTAX
        except Exception as exc:
            logger.warning(f"Error when running extracted function: {exc.__class__.__name__}: {exc}")
            return None, ErrorType.UNCLASSIFIED

        res = loc['ret']
        if not isinstance(res, (int, float)):
            logger.warning(f"The result returned by the extracted function "
                           f"({res}, type: {type(res).__name__}) is not a number")
            return None, ErrorType.NOT_A_NUMBER
        return res, AnswerPattern.CODE

    @classmethod
    def trim_response(cls, text: str) -> str:
        """'Trim' model response to the appearance of an end-of-response token - if any."""

        for stop_token in cls.STOP_TOKENS:
            idx = text.find(stop_token)
            if idx >= 0:  # -1 if not found
                return text[:idx]  # don't look for other stop tokens

        return text  # return original text if no stop tokens found
