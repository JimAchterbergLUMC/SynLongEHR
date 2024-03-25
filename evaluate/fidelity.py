# executes descriptive statistics and tsne
# note that GoF test is part of utility.py, since it uses preprocessed data
# tSNE and descriptive statistics use the raw data

import pandas as pd
import numpy as np
import os
from utils import preprocess, metrics, models
import pickle
from matplotlib import pyplot as plt
import seaborn as sns


def stats_plot(
    real_df: pd.DataFrame,
    cpar_df: pd.DataFrame,
    dgan_df: pd.DataFrame,
    result_path: str,
):
    """
    Creates plot of descriptive statistics.

    real_df: dataframe of real data.
    cpar_df: synthetic dataframe from CPAR.
    dgan_df: synthetic dataframe from DoppelGANger.
    result_path: path to save results.
    returns: None, saves plots to results directory.

    """
    barWidth = 0.25
    stats_real, stats_cpar = metrics.descriptive_statistics(real_df, cpar_df)
    stats_real, stats_dgan = metrics.descriptive_statistics(real_df, dgan_df)
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
        label="DoppelGANger",
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
    plt.savefig(os.path.join(result_path, "descr_stats_box.png"))
    plt.show()

    freqmatrix_real = metrics.rel_freq_matrix(data=real_df, columns="icd_code")
    freqmatrix_cpar = metrics.rel_freq_matrix(data=cpar_df, columns="icd_code")
    freqmatrix_dgan = metrics.rel_freq_matrix(data=dgan_df, columns="icd_code")

    # plot frequencies as a heatmap
    for matrix, name in zip(
        [freqmatrix_real, freqmatrix_cpar, freqmatrix_dgan],
        ["Real", "CPAR", "DoppelGANger"],
    ):
        plt.figure(figsize=(10, 6))
        sns.set_theme(font_scale=1.1)
        sns.heatmap(matrix, annot=False, cmap="rocket_r", fmt=".2f", vmin=0, vmax=0.8)
        plt.xlabel("Step", fontsize=11)
        plt.ylabel("Category", fontsize=11)
        plt.title(f"{name} ICD chapter frequencies", fontsize=11)
        plt.savefig(os.path.join(result_path, f"{name}_matrixplot.png"))
        plt.show()


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
    _, bins, _ = plt.hist(
        s_steps, bins="auto", color="red", label="Synthetic", alpha=0.5
    )
    plt.hist(r_steps, bins=bins, color="blue", label="Real", alpha=0.5)
    plt.xlabel("Max steps")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Maximum #steps per sample")
    return plt


# executes the tsne step
def embedding_plot(
    real_df: pd.DataFrame,
    syn_df: pd.DataFrame,
    emb_type: str = "tsne",
    n_neighbours: int = 25,
    save_distance: bool = False,
    result_path: str = "",
):
    """
    Generates embeddings from tSNE or UMAP and creates a plot.

    real_df: Real longitudinal pandas dataframe.
    syn_df: Synthetic longitudinal pandas dataframe.
    emb_type: tsne or umap embeddings.
    n_neighbours: number of neighbours used in embedding algorithm (perplexity for tSNE)
    save_distance: whether to save distance matrix to result directory. May help if we want to precompute distances and try different parameters for the embedding algorithm.
    result_path: Path to which we save (intermediary) results.
    returns: embedding plot.

    """
    assert emb_type in ["tsne", "umap"]

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

    col_len = static.shape[1] + seq.shape[2]

    # # take weighted sum of static and timevarying distances
    distance_matrix = ((len(static.columns)) / col_len) * static_distances + (
        seq.shape[2] / col_len
    ) * dynamic_distances
    if save_distance:
        filename = "distance_matrix.pkl"
        with open(os.path.join(result_path, filename), "wb") as f:
            pickle.dump(distance_matrix, f)

    # #compute and plot projections with synthetic/real labels as colors
    labels = np.concatenate(
        (np.zeros(real_df.subject_id.nunique()), np.ones(syn_df.subject_id.nunique())),
        axis=0,
    )

    if emb_type == "tsne":
        embeddings = models.tsne(distance_matrix, n_neighbours)
    else:
        embeddings = models.umap(distance_matrix, n_neighbours)
    plt.scatter(embeddings[:, 0], embeddings[:, 1], c=labels, cmap="bwr", alpha=0.1)
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="blue", label="Real"
        ),
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="red", label="Synthetic"
        ),
    ]
    plt.legend(handles=handles, fontsize=11, loc="upper right")
    plt.title(f"{emb_type} plot ({n_neighbours} neighbours)", fontsize=11)
    return plt


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

    stats_plot(real_df, cpar_df, dgan_df, result_path)

    if syn_model == "cpar":
        syn_df = cpar_df
    else:
        syn_df = dgan_df

    steps_plot = steps(real_df, syn_df)
    filename = "step_plot.png"
    steps_plot.savefig(os.path.join(result_path, filename))
    steps_plot.show()

    # select only k subjects to test code quickly
    k = 60
    syn_df = syn_df[
        syn_df.subject_id.isin(np.random.choice(syn_df.subject_id.unique(), k))
    ]
    real_df = real_df[
        real_df.subject_id.isin(np.random.choice(real_df.subject_id.unique(), k))
    ]

    emb_type = "umap"
    emb_plot = embedding_plot(real_df, syn_df, n_neighbours=25, emb_type=emb_type)
    filename = f"{emb_type} plot.png"
    emb_plot.savefig(os.path.join(result_path, filename))
    emb_plot.show()
