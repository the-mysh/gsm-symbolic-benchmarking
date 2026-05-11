import torch
from transformers import StoppingCriteria


class StopOnStringCriteria(StoppingCriteria):
    def __init__(self, stop_strings: list[str], tokenizer):
        self.stop_strings = stop_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Decode only the last few tokens to check for the stop string.
        # Decoding the whole sequence every step is too slow.
        tail_tokens = input_ids[0][-10:]
        generated_text = self.tokenizer.decode(tail_tokens, skip_special_tokens=True)

        for stop_str in self.stop_strings:
            if stop_str in generated_text:
                return True
        return False
