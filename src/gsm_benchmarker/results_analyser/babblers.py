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


def compare_babblers(b1, b2, title1, title2):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, c in enumerate(('accuracy', 'babbler percentage')):
        ax = axes[i]
        ax.set_title(f"{c.capitalize()}, %")

        for family in b2.family.unique():
            bb2 =  b2[b2['family'] == family]
            bb1 = b1[b1['family'] == family]
            bb1 = bb1[bb1.index.isin(bb2.index)]
            ax.scatter(100*bb1[c], 100*bb2[c], marker='d', label=family)

        m = 1
        lims = (-m, 100+m)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

        ax.set_aspect('equal')
        ax.axline([0, 0], [1, 1], c='k', lw=1, linestyle='--')
        ax.legend(fancybox=True, framealpha=0.5, frameon=True, title='Family')
        ax.set_xlabel(title1)
        ax.set_ylabel(title2)