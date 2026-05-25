"""Few-shot example management for prompt construction.

Provides SingleShot dataclass and GSMShotManager for loading, formatting, and
compiling few-shot examples with support for alternative solution files.
"""

from dataclasses import dataclass
from typing import Iterator

from gsm_benchmarker.utils.resources_manager import load_resource_json, load_8shot_solutions


@dataclass
class SingleShot:
    """A single few-shot example.

    Attributes:
        question: The problem question.
        solution: The solution or reasoning step(s).
        result: The numerical answer.
        sid: Shot ID (1-indexed position in the shot set).
    """
    question: str
    solution: str
    result: str
    sid: int  # shot id

    def compile(self, fmt_string: str) -> str:
        """Format this shot using a template string.

        The format string can include these placeholders:
        - {question}: The problem question
        - {solution}: The reasoning/solution
        - {result}: The numerical answer
        - {sid}: The shot ID (1-indexed)

        Args:
            fmt_string: Template string with placeholders.

        Returns:
            Formatted shot text.

        Raises:
            ValueError: If fmt_string uses undefined placeholders.
        """
        try:
            s = fmt_string.format(question=self.question, solution=self.solution, result=self.result, sid=self.sid)
        except KeyError:
            raise ValueError(
                f"The SingleShot format string may have the following fields: 'question', 'solution', 'result', "
                f"'and 'sid' (shot id). "
                f"Got:\n{fmt_string}")
        return s


class GSMShotManager:
    """Manager for a collection of few-shot examples.

    Loads 8-shot examples from resources/standard-8-shots.json, with optional
    alternative solution files for code-based or specialized prompting.
    Provides efficient compilation of shots with custom formatting.
    """
    def __init__(self, solutions_name: str | None = None, code: bool = False):
        """Initialize shot manager.

        Args:
            solutions_name: Name of alternative solutions file (e.g., 'python_functions'),
                or None to use default solutions from standard-8-shots.json.
            code: If True, interpret solutions as Python code; used when loading
                alternative solutions.
        """
        self._shots = self._load_data(solutions_name, code=code)

    @property
    def shots(self) -> tuple[SingleShot, ...]:
        """Return all loaded shots as a tuple."""
        return self._shots

    def __len__(self):
        """Return the number of available shots."""
        return len(self._shots)

    def __iter__(self) -> Iterator[SingleShot]:
        """Iterate over all shots in order."""
        return iter(self._shots)

    def __getitem__(self, item) -> SingleShot:
        """Access a shot by index."""
        return self._shots[item]

    @staticmethod
    def _load_data(solutions_file: str | None = None, code: bool = False) -> tuple[SingleShot, ...]:
        """Load shot data from resources and optional alternative solutions.

        Args:
            solutions_file: Name of alternative solutions file.
            code: If True, load solutions as code.

        Returns:
            Tuple of SingleShot objects indexed from 1.

        Raises:
            RuntimeError: If alternative solutions count doesn't match.
        """
        data_dict = load_resource_json("standard-8-shots.json")

        if solutions_file:
            # load alternative solutions
            alternative_solutions = load_8shot_solutions(solutions_file, code=code)
            if len(alternative_solutions) < len(data_dict["samples"]):
                raise RuntimeError(f"The number of alternative solutions ({len(alternative_solutions)}) "
                                   f"does not match the number of shots ({len(data_dict['samples'])})")
            for i in range(min(len(alternative_solutions), len(data_dict["samples"]))):
                data_dict["samples"][i]["solution"] = alternative_solutions[i]

        return tuple(SingleShot(**s, sid=i+1) for i, s in enumerate(data_dict["samples"]))

    def compile(self, fmt_string: str, n_shots: int | None = None, separator: str = "\n\n"):
        """Compile shots into formatted text.

        Args:
            fmt_string: Template string for formatting each shot (passed to SingleShot.compile).
            n_shots: Number of shots to include (None = all available).
            separator: String to join formatted shots.

        Returns:
            Concatenated formatted shots joined by separator.
        """
        return separator.join(s.compile(fmt_string) for s in self._shots[:n_shots])


if __name__ == '__main__':

    m = GSMShotManager()
    f = "Question:\n{question}\n\nAnswer:\n{solution}\nThe final result is: {result}"
    print(m.compile(f, n_shots=3, separator="\n\n\n"))

    print()
    print(20*"=")
    print()

    m2 = GSMShotManager(solutions_name="python_functions", code=True)
    f = "Question:\n{question}\n\nAnswer:\ndef solution():\n{solution}"
    print(m2.compile(f, n_shots=2, separator="\n\n"))

