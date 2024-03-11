# executes descriptive statistics and tsne
# note that GoF test is part of utility.py, since it uses preprocessed data
# tSNE and descriptive statistics use the raw data

import pandas as pd
import numpy as np
import os
from utils import preprocess, metrics, models
import pickle
from matplotlib import pyplot as plt


def stats_plot(real_df, cpar_df, dgan_df):
    barWidth = 0.25
    stats_real, stats_cpar = descriptive_statistics(real_df, cpar_df)
    stats_real, stats_dgan = descriptive_statistics(real_df, dgan_df)
    # boxplots
    plt.figure(figsize=(15, 4))
    box = plt.boxplot(
        real_df.age / 100, positions=[0], patch_artist=True, showfliers=False
    )
    box["boxes"][0].set_facecolor("blue")
    box = plt.boxplot(
        cpar_df.age / 100, positions=[0 + barWidth], patch_artist=True, showfliers=False
    )
    box["boxes"][0].set_facecolor("lightcoral")
    box = plt.boxplot(
        dgan_df.age / 100,
        positions=[0 + 2 * barWidth],
        patch_artist=True,
        showfliers=False,
    )
    box["boxes"][0].set_facecolor("maroon")

    # barplots
    plt.bar(
        np.arange(start=1, stop=len(stats_cpar)),
        stats_real[1:],
        color="blue",
        label="Real",
        width=barWidth,
    )
    plt.bar(
        np.arange(start=1, stop=len(stats_cpar)) + barWidth,
        stats_cpar[1:],
        color="lightcoral",
        label="CPAR",
        width=barWidth,
    )
    plt.bar(
        np.arange(start=1, stop=len(stats_cpar)) + 2 * barWidth,
        stats_dgan[1:],
        color="maroon",
        label="DGAN",
        width=barWidth,
    )

    plt.legend(fontsize=11)
    vars = [
        "Age \n /100",
        "Deceased: \n Yes",
        "Gender: \n Male",
        "Race: \n White",
        "Race: \n Unknown",
        "Race: \n Black",
        "Race: \n Hispanic",
        "Race: \n Asian",
        "Race: \n Native American",
        "Race: \n Multiple",
    ]
    plt.xticks(np.arange(len(vars)), vars, fontsize=11)
    plt.yticks(fontsize=11)
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


# def descriptive_statistics(
#     real_df: pd.DataFrame, syn_df: pd.DataFrame, result_path: str
# ):
#     """
#     Outputs descriptive statistics. Tables for static, heatmaps for dynamic data.

#     real_df: Real longitudinal pandas dataframe.
#     syn_df: Synthetic longitudinal pandas dataframe.
#     result_path: Path to which we save results.
#     returns: None, saves results to result directory.
#     """

#     static_features = ["age", "gender", "deceased", "race"]
#     # get static feature dataframes
#     real_df_static = preprocess.get_static(data=real_df, columns=static_features)
#     syn_df_static = preprocess.get_static(data=syn_df, columns=static_features)

#     # get descriptive statistics for static numerical variables
#     stat_num_real = metrics.descr_stats(data=real_df_static[["age"]])
#     stat_num_syn = metrics.descr_stats(data=syn_df_static[["age"]])
#     # filename = "descr_stats_staticnumerical.csv"
#     # pd.concat([stat_num_real, stat_num_syn], axis=1).to_csv(
#     #     os.path.join(result_path, filename)
#     # )

#     # #get relative frequencies for static categorical variables
#     rel_freq_real = metrics.relative_freq(
#         data=real_df_static[["gender", "deceased", "race"]]
#     )
#     rel_freq_syn = metrics.relative_freq(
#         data=syn_df_static[["gender", "deceased", "race"]]
#     )
#     # filename = "descr_stats_staticcategorical.csv"
#     # pd.concat([rel_freq_real, rel_freq_syn], axis=1).to_csv(
#     #     os.path.join(result_path, filename)
#     # )

#     # get matrix of relative frequencies at each step
#     # freqmatrix_real = metrics.rel_freq_matrix(data=real_df, columns="icd_code")
#     # freqmatrix_syn = metrics.rel_freq_matrix(data=syn_df, columns="icd_code")

#     # # plot frequencies as a heatmap
#     # for matrix, name in zip([freqmatrix_real, freqmatrix_syn], ["Real", "Synthetic"]):
#     #     plot = metrics.freq_matrix_plot(matrix, range=(0, 0.8))
#     #     plot.title(f"{name} ICD chapter frequencies")
#     #     filename = f"{name}_matrixplot.png"
#     #     plot.savefig(os.path.join(result_path, filename))
#     #     plot.show()
#     return stat_num_real, stat_num_syn, rel_freq_real, rel_freq_syn


def steps(real_df: pd.DataFrame, syn_df: pd.DataFrame, subject_idx: str = "subject_id"):
    """
    Outputs a plot of the amount of sequential steps across samples, to check if we accurately capture this in synthetic data.

    real_df: Real longitudinal pandas dataframe.
    syn_df: Synthetic longitudinal pandas dataframe.
    subject_idx: Subject identifier.
    returns: Plot of amount of steps per sample (real vs. synthetic).
    """

    r_steps = real_df.groupby(subject_idx).seq_num.max()
    s_steps = syn_df.groupby(subject_idx).seq_num.max()
    steps_plot = metrics.plot_max_steps(r_steps, s_steps)
    steps_plot.title("Maximum #steps per sample")
    return steps_plot


# executes the tsne step
def tsne_plot(real_df: pd.DataFrame, syn_df: pd.DataFrame, result_path: str):
    """
    Generates tSNE embeddings and creates a plot.

    real_df: Real longitudinal pandas dataframe.
    syn_df: Synthetic longitudinal pandas dataframe.
    result_path: Path to which we save (intermediary) results.
    returns: tSNE plot.

    """

    # split data into static and dynamic
    df = pd.concat([real_df, syn_df], axis=0)
    static = preprocess.get_static(df, ["age", "gender", "deceased", "race"])
    static["age"] = static["age"].astype(float)
    df["icd_code"], _ = pd.factorize(df["icd_code"])
    seq = preprocess.df_to_3d(df, cols=["icd_code"], padding=-1)

    # #find distance matrices
    static_distances = models.static_gower_matrix(
        static, cat_features=[False, True, True, True]
    )
    dynamic_distances = models.dyn_gower_matrix(seq)

    # # take weighted sum of static and timevarying distances
    distance_matrix = ((len(static.columns)) / len(df.columns)) * static_distances + (
        seq.shape[2] / len(df.columns)
    ) * dynamic_distances
    filename = "distance_matrix.pkl"
    with open(os.path.join(result_path, filename), "wb") as f:
        pickle.dump(distance_matrix, f)

    # #compute and plot tsne projections with synthetic/real labels as colors
    labels = np.concatenate(
        (np.zeros(real_df.subject_id.nunique()), np.ones(syn_df.subject_id.nunique())),
        axis=0,
    )
    tsne_plot = models.tsne(distance_matrix, labels)
    # tsne_plot.title('tSNE plot of synthetic/real samples')
    return tsne_plot


if __name__ == "__main__":
    # set the result path
    syn_model = "cpar"
    result_path = os.path.join("results", syn_model)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    # load real and synthetic data
    load_path = "data/generated"
    real_file = os.path.join(load_path, "real.csv.gz")
    syn_file = os.path.join(load_path, f"{syn_model}.csv.gz")
    cols = ["subject_id", "seq_num", "icd_code", "gender", "age", "deceased", "race"]
    real_df = pd.read_csv(real_file, sep=",", compression="gzip", usecols=cols)

    cpar_df = pd.read_csv(
        os.path.join(load_path, "cpar.csv.gz"),
        sep=",",
        compression="gzip",
        usecols=cols,
    )
    dgan_df = pd.read_csv(
        os.path.join(load_path, "dgan.csv.gz"),
        sep=",",
        compression="gzip",
        usecols=cols,
    )

    stats_plot_ = stats_plot(real_df, cpar_df, dgan_df)
    stats_plot_.savefig(os.path.join(result_path, "descr_stats_box.png"))
    stats_plot_.show()

    # if syn_model == 'cpar':
    #     syn_df = cpar_df
    # else:
    #     syn_df = dgan_df

    # select only k subjects to test code quickly
    # k = 60
    # syn_df = syn_df[
    #     syn_df.subject_id.isin(np.random.choice(syn_df.subject_id.unique(), k))
    # ]
    # real_df = real_df[
    #     real_df.subject_id.isin(np.random.choice(real_df.subject_id.unique(), k))
    # ]

    # steps_plot = steps(real_df, syn_df)
    # filename = "step_plot.png"
    # steps_plot.savefig(os.path.join(result_path, filename))
    # steps_plot.show()

    # tsne_plot_ = tsne_plot(real_df, syn_df, result_path)
    # filename = "tsne.png"
    # tsne_plot_.savefig(os.path.join(result_path, filename))
    # tsne_plot_.show()
