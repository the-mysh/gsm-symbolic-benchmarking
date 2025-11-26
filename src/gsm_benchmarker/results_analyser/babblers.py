import pandas as pd
import matplotlib.pyplot as plt

from gsm_benchmarker.results_analyser import MultiModelResultsAnalyser


def get_babbler_counts(mm: MultiModelResultsAnalyser) -> pd.DataFrame:
    babbler_examples = mm.full_data[mm.full_data.full_response.apply(lambda r: 'Q:' in r)]

    babbler_counts = babbler_examples["model"].value_counts()
    babbler_counts.name = "babbler count"

    babbler_percentage = babbler_counts / mm.full_data["model"].value_counts()
    babbler_percentage.name = "babbler percentage"

    family = mm.summary_data.index.to_series().apply(lambda v: v.split('_')[0])
    family.name = 'family'

    b = pd.concat((family, mm.summary_data, babbler_counts, babbler_percentage), axis=1)
    b.fillna(0, inplace=True)
    b.sort_values(['babbler percentage', 'accuracy'], ascending=False)

    return b


def plot_babblers_by_family(b):
    fig, ax = plt.subplots()
    ax.set_ylabel("accuracy, %")
    ax.set_xlabel("babbler factor, %")
    for family in b['family'].unique():
        bb =  b[b['family'] == family]
        ax.scatter(100*bb['babbler percentage'], 100*bb['accuracy'], marker='d', label=family)
    ax.legend(fancybox=True, framealpha=0.5, frameon=True, title='Family')
    ax.set_aspect('equal')

    m = 1
    lims = (-m, 100+m)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    return fig