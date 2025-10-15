from importlib.resources import files
import json
from dataclasses import dataclass
from typing import Iterator


_RESOURCES_PATH = files("gsm_benchmarker")/"resources"


@dataclass()
class SingleShot:
    question: str
    solution: str
    result: str

    def format(self, fmt_string: str) -> str:
        try:
            s = fmt_string.format(question=self.question, solution=self.solution, result=self.result)
        except KeyError:
            raise ValueError(
                f"The SingleShot format string should have fields: 'question', 'solution', and 'result'. "
                f"Got:\n{fmt_string}")
        return s



class GSM8hotManager:
    def __init__(self):
        self._shots = self._load_data()

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
    def _load_data() -> tuple[SingleShot, ...]:
        data_bytes = (_RESOURCES_PATH / "standard-8-shots.json").read_bytes()
        data_dict = json.loads(data_bytes)
        return tuple(SingleShot(**s) for s in data_dict["samples"])

    def format(self, fmt_string: str, n_shots: int | None = None, separator: str = "\n\n"):
        return separator.join(s.format(fmt_string) for s in self._shots[:n_shots])


if __name__ == '__main__':
    m = GSM8hotManager()

    f = "Question:\n{question}\n\nAnswer:\n{solution}\nThe final result is: {result}"
    print()
    print(m.format(f, n_shots=3, separator="\n\n\n"))


