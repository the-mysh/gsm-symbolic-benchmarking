import os
import pandas as pd
from pathlib import Path
import logging

from gsm_benchmarker.benchmark.answer_extractor import AnswerExtractor


logging.getLogger('gsm_benchmarker').setLevel(logging.ERROR)

RESULTS_FOLDER = 'mini_20x50x4__14_11'
CODE = False


ROOT_PATH = Path(f"{__file__}/../../../../data/gsm-symbolic/outputs").resolve()
p_results = ROOT_PATH / RESULTS_FOLDER / 'final'
print(f"TARGET PATH: {p_results}")

p_results_corrected = p_results.parent / 'corrected'
print(f"CORRECTED RESULTS PATH: {p_results_corrected}")
os.makedirs(p_results_corrected, exist_ok = True)

print("FORMAT: " + ('CODE' if CODE else 'TEXTUAL'))


all_fixed = 0

for folder in os.listdir(p_results):
    if not (p_results / folder).is_dir():
        continue
    print("\n\n" + folder + "\n-------")
    os.makedirs(p_results_corrected / folder, exist_ok = True)

    for model_pq in os.listdir(p_results / folder):
        print(model_pq)
        m = pd.read_parquet(p_results / folder / model_pq)
        print(f"\tall errors    : {m.shape[0] - m.correct.sum()}")

        fixed = 0
        for i in m.index:
            row = m.loc[i]

            if CODE:
                result, result_type = AnswerExtractor.extract_answer_code(row.full_response)
            else:
                result, result_type = AnswerExtractor.extract_answer_textual(row.full_response)

            m.loc[i, "detected_result_pattern"] = result_type.name

            if result == m.loc[i, "predicted_numerical_result"]:
                continue  # no change

            fixed += 1
            m.loc[i, "predicted_numerical_result"] = result
            m.loc[i, "correct"] = (result == m.loc[i, "numerical_result"])

        all_fixed += fixed
        print(f"\tfixed         : {fixed}")
        print(f"\tall errors now: {m.shape[0] - m.correct.sum()}")
        print("\n")
        m.to_parquet(p_results_corrected / folder / model_pq)



print(f"Fixed {all_fixed} answers")
