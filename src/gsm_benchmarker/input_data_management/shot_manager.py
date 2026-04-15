from dataclasses import dataclass
from typing import Iterator
from pathlib import Path

from gsm_benchmarker.utils.resources_manager import load_resource_json, load_8shot_solutions


@dataclass
class SingleShot:
    question: str
    solution: str
    result: str
    sid: int  # shot id

    def compile(self, fmt_string: str) -> str:
        try:
            s = fmt_string.format(question=self.question, solution=self.solution, result=self.result, sid=self.sid)
        except KeyError:
            raise ValueError(
                f"The SingleShot format string may have the following fields: 'question', 'solution', 'result', "
                f"'and 'sid' (shot id). "
                f"Got:\n{fmt_string}")
        return s


class GSMShotManager:
    def __init__(self, solutions_file: str | None = None, code: bool = False):
        self._shots = self._load_data(solutions_file, code=code)

    @property
    def shots(self) -> tuple[SingleShot, ...]:
        return self._shots

    def __len__(self):
        return len(self._shots)

    def __iter__(self) -> Iterator[SingleShot]:
        return iter(self._shots)

    def __getitem__(self, item) -> SingleShot:
        return self._shots[item]

    @staticmethod
    def _load_data(solutions_file: str | None = None, code: bool = False) -> tuple[SingleShot, ...]:
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
        return separator.join(s.compile(fmt_string) for s in self._shots[:n_shots])


if __name__ == '__main__':

    m = GSMShotManager()
    f = "Question:\n{question}\n\nAnswer:\n{solution}\nThe final result is: {result}"
    print(m.compile(f, n_shots=3, separator="\n\n\n"))

    print()
    print(20*"=")
    print()

    m2 = GSMShotManager(solutions_file="python_8shot_solutions.py", code=True)
    f = "Question:\n{question}\n\nAnswer:\ndef solution():\n{solution}"
    print(m2.compile(f, n_shots=2, separator="\n\n"))

