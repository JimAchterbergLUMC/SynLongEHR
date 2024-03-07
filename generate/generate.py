import os
import pickle
import utils.preprocess as preprocess
import pandas as pd


def generate_cpar(n_samples: int):
    """
    Generates synthetic samples from trained CPAR model.

    n_samples: amount of synthetic subjects to generate
    returns: synthetic samples
    """
    load_path = os.path.join("syn_model", "cpar.pkl")
    model = pickle.load(open(load_path, "rb"))
    samples = model.sample(num_sequences=n_samples, sequence_length=None)
    return samples


def generate_dgan(n_samples: int):
    """
    Generates synthetic samples from trained DGAN model.

    n_samples: amount of synthetic subjects to generate
    returns: synthetic samples
    """
    model_path = os.path.join("syn_model", "dgan_model.pkl")
    weight_path = os.path.join("syn_model", "dgan_weights.pt")
    # load instantiated model and input saved weights
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    model = model.load(weight_path)
    # return samples
    samples = model.generate_dataframe(n_samples)
    return samples


def generate_noise(
    df: pd.DataFrame,
    cont_noise_lvl: float = 0.1,
    cat_noise_lvl: float = 0.1,
    cat_cols: list = ["gender", "deceased", "race", "icd_code"],
    cont_cols: list = ["age"],
):
    """
    Generates synthetic samples from noise model.

    df: pandas dataframe of real data
    n_samples: amount of synthetic subjects to generate
    cont_noise_lvl: standard deviation of Gaussian noise added to continuous features
    cat_noise_lvl: perturbation probability of categorical features
    cat_cols: categorical features
    cont_cols: continuous features
    returns: synthetic samples
    """
    samples = df.copy()
    samples[cat_cols] = samples[cat_cols].apply(
        preprocess.rd_perturbation, level=cat_noise_lvl
    )
    samples[cont_cols] = samples[cont_cols].apply(
        preprocess.rd_noise, level=cont_noise_lvl
    )
    return samples


if __name__ == "__main__":
    n_samples = 100
    syn_model = "cpar"  # change to 'cpar' or 'noise' if required
    assert syn_model in ["dgan", "cpar", "noise"]

    if syn_model == "dgan":
        samples = generate_dgan(n_samples)
        samples = preprocess.postprocess_dgan(samples)
    elif syn_model == "cpar":
        samples = generate_cpar(n_samples)
        samples = preprocess.postprocess_cpar(samples)
    else:
        df = pd.read_csv("data/generated/real.csv.gz", compression="gzip", sep=",")
        samples = generate_noise(
            df,
            cont_noise_lvl=0.1,
            cat_noise_lvl=0.1,
            cat_cols=["gender", "deceased", "race", "icd_code"],
            cont_cols=["age"],
        )

    save_path = os.path.join("data", "generated")
    file = f"{syn_model}.csv.gz"
    samples.to_csv(os.path.join(save_path, file), compression="gzip", sep=",")
