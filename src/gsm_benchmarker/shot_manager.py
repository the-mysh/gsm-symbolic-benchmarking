from importlib.resources import files
import json
from dataclasses import dataclass
from typing import Iterator

_RESOURCES_PATH = files("gsm_benchmarker")/"resources"


@dataclass()
class SingleShot:
    question: str
    answer: str


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


if __name__ == '__main__':
    m = GSM8hotManager()
    print()
    for i, ex in enumerate(m):
        if i == 3:
            break
        print(f"Question {i}: {ex.question}")
        print(f"Answer {i}: {ex.answer}")
        print()

    print("...")


