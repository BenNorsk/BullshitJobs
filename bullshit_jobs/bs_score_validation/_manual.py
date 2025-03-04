from bullshit_jobs.load_data._load_data import _quick_load, _save_data
import pandas as pd
import numpy as np


def _select_random_sample() -> pd.DataFrame:
    """
    Selects a random sample of the data.

    Returns:
    --------
    pd.DataFrame
        The random sample.
    """
    # Set a seed of 42 for reproducibility
    np.random.seed(42)
    df = _quick_load("data_with_bs_scores.pkl")

    # Turn the "bs_score_llm" column into a float
    df["bs_score"] = df["bs_score_llm"].astype(float)

    # Make a sample_df
    sample_df = None

    for i in range(11):
        print(i)
        df_selected = df[df["bs_score_llm"] == (i / 10)]
        print(df_selected)
        if i == 0:
            sample = df_selected.sample(40)
        elif i == 10:
            sample = df_selected.sample(10)
        else:
            sample = df_selected.sample(20)

        sample_df = pd.concat([sample_df, sample])

    print(sample_df)

    # Print a description of the columns bs_score_llm and bs_score_binary_dict
    print(sample_df["bs_score_llm"].describe())

    # Print a description of the columns bs_score_llm and bs_score_binary_dict
    print(sample_df["bs_score_binary_dict"].describe())

    # Shuffle the data set
    sample_df = sample_df.sample(frac=1)

    # Drop all columns except the following
    # cons, review_id
    sample_df = sample_df[["cons", "review_id"]]

    # Save the sample df
    _save_data(sample_df, "manual_evaluation_task")

    return sample_df

if __name__ == "__main__":
    df = _select_random_sample()
    print(df)