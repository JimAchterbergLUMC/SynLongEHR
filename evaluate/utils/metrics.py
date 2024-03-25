from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_absolute_error,
    accuracy_score,
    roc_auc_score,
)
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import ks_2samp
from utils import preprocess


def mape(true, pred):
    return mean_absolute_percentage_error(true, pred)


def mae(true, pred):
    return mean_absolute_error(true, pred)


def accuracy(true, pred):
    return accuracy_score(true, pred)


def auc(labels, pred_scores):
    return roc_auc_score(labels, pred_scores)


def ks_test(real_pred, syn_pred):
    return ks_2samp(
        data1=real_pred.flatten(), data2=syn_pred.flatten(), alternative="two-sided"
    )


def rel_freq_matrix(data, columns, timestep_idx="seq_num"):
    rel_freq = (
        data.groupby(timestep_idx)[columns]
        .value_counts(normalize=True)
        .rename("rel_freq")
        .reset_index()
    )
    rel_freq = rel_freq.pivot(index=columns, columns=timestep_idx, values="rel_freq")
    return rel_freq


def GoF_kdeplot(pred, y_test):
    plt.figure(figsize=(10, 6))
    range = (0, 1)
    plt.xlim(0, 1)
    plt.ylim(0, 6)
    sns.kdeplot(
        pred[y_test == 0],
        palette=["blue"],
        clip=range,
        alpha=0.5,
        fill=True,
        label="Real",
    )
    sns.kdeplot(
        pred[y_test == 1],
        palette=["red"],
        clip=range,
        alpha=0.5,
        fill=True,
        label="Synthetic",
    )
    plt.xlabel("Classification score", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(fontsize=11)
    return plt


def descriptive_statistics(real_df: pd.DataFrame, syn_df: pd.DataFrame):
    """
    Creates a list of some basic descriptive statistics, to later be plotted in a Bar plot.

    real_df: real dataframe
    syn_df: synthetic dataframe
    returns: list of descriptive statistics
    """

    # first get static data
    stat = ["age", "gender", "deceased", "race"]
    real_df = preprocess.get_static(data=real_df, columns=stat)
    syn_df = preprocess.get_static(data=syn_df, columns=stat)

    stats_real = []
    stats_syn = []
    for df, results in zip([real_df, syn_df], [stats_real, stats_syn]):
        results.append(df.age.mean() / 100)
        results.append(df.deceased.mean())
        results.append(len(df[df.gender == "M"]) / len(df))
        for race in [
            "white",
            "unknown",
            "black",
            "hispanic",
            "asian",
            "native_american",
            "multiple",
        ]:
            results.append(len(df[df.race == race]) / len(df))
    return stats_real, stats_syn
