"""Extract numeric answers from GSM-Symbolic model responses.

The extractor supports two modes:

* textual extraction, which looks for common answer phrases and falls back to
  the last number in the response; and
* code extraction, which parses a generated function, executes it in a
  restricted namespace, and interprets the return value as the answer.

The code-execution path intentionally uses a small allowlist of builtins and
imports plus a few string-level safety checks to reduce the risk of running
dangerous generated code.
"""

import re
import logging
from enum import Enum, auto
from typing import Any


logger = logging.getLogger(__name__)


class AnswerPattern(Enum):
    """Successful answer-extraction patterns."""

    GMS8K = auto()
    GSM_SYMBOLIC = auto()
    LAST_NUMBER = auto()
    CODE = auto()



class ErrorType(Enum):
    """Failure modes encountered while extracting or executing an answer."""

    NO_NUMBER = auto()  # for textual answers - when a number could not be extracted
    NO_FUNCTION = auto()  # failed to extract function definition
    SYNTAX_ERROR = auto()  # function definition extracted, but has invalid syntax
    NAME_ERROR = auto()  # name error encountered when running the function
    FORBIDDEN_STRING = auto()  # one of the potentially dangerous strings (e.g. 'eval') found in function
    NONE_RETURNED = auto()  # for code answers - when function returns None
    NOT_A_NUMBER = auto()  # for code answers - when the return value of a function is not a number (and not None)
    TYPE_VALUE_ERROR = auto()  # TypeError or ValueError
    ZERO_DIVISION_ERROR = auto()
    ATTRIBUTE_ERROR = auto()
    UNCLASSIFIED = auto()  # all others
    EMPTY_RESPONSE = auto()  # for textual answers - when empty, before trimming
    EMPTY_AFTER_TRIM = auto()


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
    """Extract answers from either textual model outputs or generated code."""

    _number_pattern_str = r'-?\d+(?:\.\d+)?'
    _import_pattern = re.compile(r'\s*(from \w+ )?import \w+')

    ANSWER_PATTERNS = {
        AnswerPattern.GMS8K: re.compile(r'####\s*[^\d-]*(' + _number_pattern_str + ')'),
        AnswerPattern.GSM_SYMBOLIC: re.compile(r'[Tt]he (?:final )?answer is:?\s*[^\d-]*(' + _number_pattern_str + ')'),
    }

    FUNCTION_PATTERN = re.compile(r"^def (?P<func_name>\w+)\(\):\s*\n(?P<body>(?:\s+.*|\n)+)", flags=re.MULTILINE)

    FORBIDDEN_ITEMS = [
        re.compile(r"open\(.*\)"),
        re.compile(r"eval\(.*\)"),
        re.compile(r"exec\(.*\)"),
        re.compile(r"__import__\(.*\)"),
        re.compile(r"[gs]etattr\(.*\)")
    ]

    BABBLER_TOKENS = ("Q:", "Question:")  # when model moves on to generating a next question

    def __init__(self, code: bool = False):
        """Create an extractor configured for textual or code-based answers."""

        self._code = code

    def extract_answer(self, text: str) -> tuple[float | int | None, AnswerPattern | ErrorType | None]:
        """Extract a numeric answer and report the extraction pattern or error."""

        res, answer_pattern_or_error_type = (
            self.extract_answer_code(text) if self._code else self.extract_answer_textual(text)
        )
        if res is None:
            logger.warning(f"-> Could not extract answer from model response:\n{text}")
        return res, answer_pattern_or_error_type

    @classmethod
    def extract_answer_textual(cls, text: str) -> tuple[float | int | None, AnswerPattern | ErrorType]:
        """Extract a numeric answer from plain-text model output.

        The extractor trims any babbler suffix, checks a small set of common
        answer formats, and finally falls back to the last number found in the
        response.
        """

        if not text:
            logger.warning("The response is empty")
            return None, ErrorType.EMPTY_RESPONSE

        text = cls.trim_response(text)

        if not text.strip('\n').strip():
            logger.warning("The response is empty after trimming generated questions")
            return None, ErrorType.EMPTY_AFTER_TRIM

        # Remove commas before regex evaluation to handle thousands (e.g. 5,000 -> 5000)
        clean_text = text.replace(',', '')

        # Try standard patterns
        for pattern_enum, pattern in cls.ANSWER_PATTERNS.items():
            match = pattern.search(clean_text)
            if match:
                # match.group(1) specifically grabs the digits, ignoring any '$' or spaces caught by [^\d-]*
                return float(match.group(1)), pattern_enum

        # Last resort: find last number in text
        numbers = re.findall(cls._number_pattern_str, clean_text)
        if numbers:
            return float(numbers[-1]), AnswerPattern.LAST_NUMBER

        logger.warning(f"Could not locate numerical answer")
        return None, ErrorType.NO_NUMBER

    @classmethod
    def check_extracted_func(cls, func_def: str):
        """Return True when a candidate function contains a forbidden construct."""

        for s in cls.FORBIDDEN_ITEMS:
            if (m := s.search(func_def)) is not None:
                logger.warning(f"Potentially dangerous string ('{m.group()}') found in the extracted function")
                return True
        return False

    @classmethod
    def extract_function_definition(cls, text: str) -> tuple[str, str]:
        """Extract a function definition from generated code-like text.

        If the model only produced a function body, a `def solution():` header
        is prepended before parsing. Import lines are removed because the code
        execution environment already exposes a safe import allowlist, and any
        post-function text is discarded.
        """

        text = cls.trim_response(text)

        text = "def solution():\n" + text

        match = cls.FUNCTION_PATTERN.search(text)
        if not match:
            return "", ""

        # remove 'import' lines - use predefined imports in local env
        lines = match.group().split('\n')
        for i in range(len(lines)):
            if cls._import_pattern.match(lines[i]):
                lines[i] = ""

        # discard post-function lines
        clean_lines = [lines[0]]
        for line in lines[1:]:
            # If the line has content and DOES NOT start with whitespace, we've hit the end of the function.
            if line.strip() and not line.startswith((' ', '\t')):
                break
            clean_lines.append(line)

        text = "\n".join(clean_lines)

        return text, match.group('func_name')

    @classmethod
    def extract_answer_code(cls, text: str) -> tuple[float | int | None, AnswerPattern | ErrorType]:
        """Extract an answer by parsing and executing generated code."""

        func_def, func_name = cls.extract_function_definition(text)

        res, answer_pattern_or_error_type, issue = cls.try_running_function(func_def, func_name)

        if issue:
            logger.warning(issue)

        return res, answer_pattern_or_error_type

    @classmethod
    def try_running_function(cls, func_def: str, func_name: str):
        """Run an extracted function and normalize non-numeric outcomes."""

        if not func_def:
            return None, ErrorType.NO_FUNCTION, "Failed to find valid function definition in text"

        res, answer_pattern_or_error_type, issue = cls.run_extracted_function(func_def, func_name=func_name)

        if isinstance(answer_pattern_or_error_type, AnswerPattern) and not isinstance(res, (int, float)):
            if res is None:
                return None, ErrorType.NONE_RETURNED, "The function did not return any value"
            else:
                issue = (f"The result returned by the extracted function "
                          f"({res}, type: {type(res).__name__}) is not a number")
                return None, ErrorType.NOT_A_NUMBER, issue

        return res, answer_pattern_or_error_type, issue

    @classmethod
    def run_extracted_function(cls, func_def: str, func_name: str = 'solution') -> tuple[Any, AnswerPattern | ErrorType, str]:
        """Execute extracted code in a restricted namespace.

        The allowlisted namespace is intentionally small. Syntax and runtime
        errors are mapped to `ErrorType` values so callers can report them in a
        structured way.
        """

        if cls.check_extracted_func(func_def):
            return None, ErrorType.FORBIDDEN_STRING, "Extracted function uses a forbidden string"

        scope = {'__builtins__': SAFE_BUILTINS.copy(), **SAFE_IMPORTS}
        loc = {}
        code = f"{func_def}\nret = {func_name}()"
        try:
            exec(code, scope, loc)
        except SyntaxError as exc:
            return None, ErrorType.SYNTAX_ERROR, f"Extracted function definition has invalid syntax: {exc}"
        except NameError as exc:
            return None, ErrorType.NAME_ERROR, f"NameError when running extracted function: {exc}"
        except AttributeError as exc:
            return None, ErrorType.ATTRIBUTE_ERROR, f"AttributeError when running extracted function: {exc}"
        except ZeroDivisionError as exc:
            return None, ErrorType.ZERO_DIVISION_ERROR, f"ZeroDivisionError when running extracted function: {exc}"
        except (TypeError, ValueError) as exc:
            return None, ErrorType.TYPE_VALUE_ERROR, f"{exc.__class__.__name__} when running extracted function: {exc}"
        except Exception as exc:
            return None, ErrorType.UNCLASSIFIED, f"Error when running extracted function: {exc.__class__.__name__}: {exc}"

        res = loc['ret']
        return res, AnswerPattern.CODE, ""

    @classmethod
    def trim_response(cls, text: str) -> str:
        """Truncate a model response at the first babbler token, if present."""

        for bt in cls.BABBLER_TOKENS:
            idx = text.find(bt)
            if idx >= 0:  # -1 if not found
                return text[:idx]  # don't look for other stop tokens

        return text  # return original text if no stop tokens found
