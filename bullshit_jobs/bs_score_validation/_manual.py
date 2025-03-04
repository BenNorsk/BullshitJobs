from bullshit_jobs.load_data._load_data import _quick_load, _save_data
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


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

def _evaluate_bs_scores_manually() -> pd.DataFrame:
    """
    Evaluates the bs_scores manually.

    Returns:
    --------
    pd.DataFrame
        The evaluated DataFrame.
    """
    # Load the data
    df = _quick_load("data_with_bs_scores.pkl")
    evaluation = pd.read_excel("./data/manual_evaluation/manual_evaluation_task_maike_filled_in.xlsx")
    print(evaluation)

    # Merge the data on review_id
    df = pd.merge(df, evaluation, on="review_id")

    # Drop all columns where score_maike is NaN
    df = df.dropna(subset=["score_maike"])
    print(df)

    # Show the correlation between score_maike and bs_score_llm
    print(df["score_maike"].corr(df["bs_score_llm"]))
    print(df["score_maike"].corr(df["bs_score_binary_dict"]))

    y = df[["score_maike"]]
    X = df[["bs_score_llm"]]

    # Fit the linear regression model
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    # Print a model summary
    print(results.summary())

    # Fit the linear regression model for bs_score_binary_dict
    y = df[["score_maike"]]
    X = df[["bs_score_binary_dict"]]
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    # Print a model summary
    print(results.summary())

    # Make a scatter plot with the linear regression line
    plt.scatter(df["bs_score_binary_dict"], df["score_maike"])
    plt.plot(df["bs_score_binary_dict"], results.predict(X), color="red")
    plt.xlabel("bs_score_binary_dict")
    plt.ylabel("score_maike")
    plt.savefig("./data/manual_evaluation/bs_score_binary_dict_vs_score_maike.png")

    return

if __name__ == "__main__":
    # df = _select_random_sample()
    # print(df)
    _evaluate_bs_scores_manually()