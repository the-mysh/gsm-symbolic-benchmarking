import os
import pandas as pd
from pathlib import Path
import logging

from gsm_benchmarker.benchmark.answer_extractor import AnswerExtractor


logging.getLogger('gsm_benchmarker').setLevel(logging.ERROR)


ROOT_PATH = Path(f"{__file__}/../../../../data/gsm-symbolic/outputs").resolve()
P_CODE = ROOT_PATH / 'code_full__09_02/final'
print(f"TARGET PATH: {P_CODE}")

p_code_corrected = P_CODE.parent / 'corrected'
print(f"CORRECTED RESULTS PATH: {p_code_corrected}")
os.makedirs(p_code_corrected, exist_ok = True)

all_fixed = 0

for folder in os.listdir(P_CODE):
    if not (P_CODE/folder).is_dir():
        continue
    print("\n\n" + folder + "\n-------")
    os.makedirs(p_code_corrected / folder, exist_ok = True)

    for model_pq in os.listdir(P_CODE/folder):
        print(model_pq)
        m = pd.read_parquet(P_CODE/folder/model_pq)
        m_se = m[m.detected_result_pattern == 'SYNTAX_ERROR']
        print(f"\tall errors    : {m.shape[0] - m.correct.sum()}")
        print(f"\tsyntax errors : {m_se.shape[0]}")
        if not m_se.shape[0]:
            continue

        fixed = 0
        for i in m_se.index:
            row = m_se.loc[i]
            result, result_type = AnswerExtractor.extract_answer_code(row.full_response)
            if result is None:
                continue
            fixed += 1
            m.loc[i, "predicted_numerical_result"] = result
            m.loc[i, "detected_result_pattern"] = result_type.name
            m.loc[i, "correct"] = (result == m.loc[i, "numerical_result"])

        all_fixed += fixed
        m.to_parquet(p_code_corrected/folder/model_pq)

        print(f"\tfixed         : {fixed}")
        print(f"\tall errors now: {m.shape[0] - m.correct.sum()}")
        print("\n")


print(f"Fixed {all_fixed} answers")
