import re
import logging


logger = logging.getLogger(__name__)


class AnswerExtractor:
    _number_pattern = r'\s*(?P<number>(-?\s?\d+(?:\.\d+)?))'

    ANSWER_PATTERNS = (
        (
            "GSM8K standard format: '#### <number>",
            re.compile(r'####' + _number_pattern)
        ),
        (
            "GSM-Symbolic format: 'The final answer is <number>'",
            re.compile(r'[Tt]he (?:final )?answer is\s*\$?' + _number_pattern)
        ),
        (
            "'Answer:' format",
            re.compile(r'[Aa]nswer:\s*\$?' + _number_pattern)
        ),
        (
            "'= <number> format'",
            re.compile(r'=' + _number_pattern)
        ),
    )

    STOP_TOKENS = (
        # from paper
        "Q:",  # model moves on to generating a next question
        "</s>",
        "<|endoftext|>",

        # other, suggested by Gemini
        "**`<",  # OpenAI / Mistral / LLama3
        "<end_of_turn>",  # Gemma
        "[/INST]"  # Mistral Dialogic / Tool Use
    )

    @classmethod
    def extract_answer(cls, text: str) -> float | None:
        """
        Extract numerical answer from text.
        Looks for patterns like "#### NUMBER" or "The (final) answer is NUMBER"
        """

        text = cls.trim_response(text)

        for pattern_name, pattern in cls.ANSWER_PATTERNS:
            match = pattern.search(text)
            if match:
                logger.debug(f"Matched answer pattern: {pattern_name}")
                return float(match.group('number'))

        logger.debug("No predefined answer pattern matched")

        # Last resort if none of the patterns work: find last number in text
        numbers = re.findall(cls._number_pattern, text)
        if numbers:
            logger.debug("Extracting answer as the last number in text")
            return float(numbers[-1][0])

        logger.warning("Could not extract answer from text")

        return None

    @classmethod
    def trim_response(cls, text: str) -> str:
        """'Trim' model response to the appearance of an end-of-response token - if any."""

        for stop_token in cls.STOP_TOKENS:
            idx = text.find(stop_token)
            if idx >= 0:  # -1 if not found
                return text[:idx]  # don't look for other stop tokens

        return text  # return original text if no stop tokens found
