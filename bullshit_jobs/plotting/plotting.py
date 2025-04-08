import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression


def _run_ols_regression(y, x):
    """
    Runs an OLS regression and returns the fitted model.
    
    Parameters:
    -----------
    y: pd.Series
        The dependent variable.
    x: pd.Series
        The independent variable.

    Returns:
    --------
    sm.OLS
        The fitted OLS model.
    """
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model


import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import Counter


def plot_simple_uniform_instrument_validation(y, x, x_1, x_2):
    fig, ax = plt.subplots()

    # Convert to numpy arrays
    x = np.array(x)
    y = np.array(y)
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)

    # Compute frequency of each (x, y) pair
    xy_pairs = list(zip(x, y))
    freq_counter = Counter(xy_pairs)
    size = np.array([freq_counter[(xi, yi)] * 20 for xi, yi in zip(x, y)])  # base size is 20 per count

    # Prepare sets for (x₁, y) and (x₂, y)
    valid_x1 = ~np.isnan(x_1)
    valid_x2 = ~np.isnan(x_2)
    x1_set = set(zip(x_1[valid_x1], y[valid_x1]))
    x2_set = set(zip(x_2[valid_x2], y[valid_x2]))

    # Determine color groupings
    in_x1 = np.array([(xi, yi) in x1_set for xi, yi in zip(x, y)])
    in_x2 = np.array([(xi, yi) in x2_set for xi, yi in zip(x, y)])
    in_both = in_x1 & in_x2
    only_x1 = in_x1 & ~in_both
    only_x2 = in_x2 & ~in_both

    # Plot colored points with size based on frequency
    ax.scatter(x[only_x1], y[only_x1], color='#95C591', label='Only in x₁', s=size[only_x1])
    ax.scatter(x[only_x2], y[only_x2], color='#2B3674', label='Only in x₂', s=size[only_x2])
    ax.scatter(x[in_both], y[in_both], color='#36868D', label='In both x₁ & x₂', s=size[in_both])

    # Fit and plot main regression line (x on y)
    reg_all = LinearRegression().fit(x.reshape(-1, 1), y)
    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_pred_all = reg_all.predict(x_range)
    ax.plot(x_range, y_pred_all, label='Regression: x on y', linestyle='--', color='black')

    # Regression for x₁ (drop NaNs)
    if np.any(valid_x1):
        reg_x1 = LinearRegression().fit(x_1[valid_x1].reshape(-1, 1), y[valid_x1])
        y_pred_x1 = reg_x1.predict(x_range)
        ax.plot(x_range, y_pred_x1, label='x₁ on y', linestyle='-', color='#95C591')

    # Regression for x₂ (drop NaNs)
    if np.any(valid_x2):
        reg_x2 = LinearRegression().fit(x_2[valid_x2].reshape(-1, 1), y[valid_x2])
        y_pred_x2 = reg_x2.predict(x_range)
        ax.plot(x_range, y_pred_x2, label='x₂ on y', linestyle='-', color='#2B3674')

    # Labels and legend
    ax.set_xlabel("x")
    ax.set_ylabel("y (bs_score_llm)")
    ax.set_title("Instrument Validation with Source-Based Coloring and Frequency Scaling")
    ax.legend()
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    # Load the data
    df = pd.read_pickle("data/master_data.pkl")
    df = df[df["bs_score_validated_uniform"].notnull()]

    # Define the bs_score_binary_dict and bs_score_llm columns
    bs_score_binary_dict = df["bs_score_binary_dict"]
    bs_score_llm = df["bs_score_llm"]
    x_1 = df["bs_score_validator_1"] # colour = #95C591
    x_2 = df["bs_score_validator_2"] # colour = #2B3674
    x = df["bs_score_validated_uniform"] # Joint colour: #36868D
    
    # Run a linear regression
    model = _run_ols_regression(bs_score_llm, x)

    # Plot the uniform instrument validation
    plot_simple_uniform_instrument_validation(bs_score_llm, x, x_1, x_2)