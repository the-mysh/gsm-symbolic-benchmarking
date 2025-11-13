from dataclasses import dataclass

from gsm_benchmarker.shot_manager import GSMShotManager


@dataclass
class PromptConfig:
    n_shots: int = 8
    question_format: str = "Q: {question}\nA: Let's think step by step."
    answer_format: str = " {solution} The final answer is {result}."
    intro: str = "As an expert problem solver, solve step by step the following mathematical questions."
    target_intro: str = ""
    separator = "\n\n"

    def __post_init__(self):
        if '{question}' not in self.question_format:
            raise ValueError("question_format must contain '{question}' placeholder")

        if '{solution}' not in self.answer_format:
            raise ValueError("answer_format must contain '{solution}' placeholder")

    @property
    def shot_format(self) -> str:
        return self.question_format + self.answer_format

    def __call__(self, question: str, shots: GSMShotManager) -> str:

        prompt = self.intro
        prompt += self.separator
        prompt += shots.compile(self.shot_format, n_shots=self.n_shots, separator=self.separator)
        prompt += self.separator
        prompt += self.question_format.format(question=question)

        return prompt
