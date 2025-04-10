import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from collections import Counter
from matplotlib.lines import Line2D


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


def plot_simple_uniform_instrument_validation(y, x, x_1, x_2):
    fig, ax = plt.subplots()

    # Set global font
    mpl.rcParams['font.family'] = 'Futura'

    # Convert inputs to numpy arrays
    x = np.array(x)
    y = np.array(y)
    x_1 = np.array(x_1)
    x_2 = np.array(x_2)

    # Frequency count of each (x, y) pair
    xy_pairs = list(zip(x, y))
    freq_counter = Counter(xy_pairs)
    size = np.array([freq_counter[(xi, yi)] * 12 for xi, yi in zip(x, y)])

    # Determine valid entries for x₁ and x₂
    valid_x1 = ~np.isnan(x_1)
    valid_x2 = ~np.isnan(x_2)
    x1_set = set(zip(x_1[valid_x1], y[valid_x1]))
    x2_set = set(zip(x_2[valid_x2], y[valid_x2]))

    # Plot 'x' markers for observations in x₁
    for xi, yi in x1_set:
        mask = (x == xi) & (y == yi)
        ax.scatter(
            x[mask], y[mask],
            marker='x',
            s=size[mask],
            color='#a0a0a0',
            edgecolors='black',
            linewidths=0.5
        )

    # Plot 'o' markers for observations in x₂
    for xi, yi in x2_set:
        mask = (x == xi) & (y == yi)
        ax.scatter(
            x[mask], y[mask],
            marker='o',
            s=size[mask],
            facecolors='none',
            edgecolors='#a0a0a0',
            linewidths=0.5
        )

    # Fit and plot main regression line (x on y)
    reg_all = LinearRegression().fit(x.reshape(-1, 1), y)
    x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
    y_pred_all = reg_all.predict(x_range)
    ax.plot(x_range, y_pred_all, label='Regression Line', linestyle='-', color='black')

    # Regression for x₁
    if np.any(valid_x1):
        reg_x1 = LinearRegression().fit(x_1[valid_x1].reshape(-1, 1), y[valid_x1])
        y_pred_x1 = reg_x1.predict(x_range)
        ax.plot(x_range, y_pred_x1, label='Regression for Validator 1', linestyle='--', color='#4997CD')

    # Regression for x₂
    if np.any(valid_x2):
        reg_x2 = LinearRegression().fit(x_2[valid_x2].reshape(-1, 1), y[valid_x2])
        y_pred_x2 = reg_x2.predict(x_range)
        ax.plot(x_range, y_pred_x2, label='Regression for Validator 2', linestyle='--', color='#CA3F38')

    # Axis labels with Futura
    ax.set_xlabel("Bullshit Score Assigned by the Validators", fontname='Futura')
    ax.set_ylabel("Bullshit Score Assigned by the LLM", fontname='Futura')

    # Make sure tick labels use Futura as well
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Futura')
        label.set_fontsize(8)

    # Custom legend entries for the data points, added last
    custom_legend = [
        Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=8, label='Data Point of Validator 1'),
        Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=8, markerfacecolor='none', label='Data Point of Validator 2')
    ]

    ax.legend(handles=ax.get_legend_handles_labels()[0] + custom_legend, fontsize=10)
    plt.tight_layout()

    # Save the figure
    plt.savefig("figures/instrument_validation/uniform_instrument_validation.png", dpi=300, bbox_inches='tight')
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