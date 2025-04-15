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
from wordcloud import WordCloud
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap


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
        ax.plot(x_range, y_pred_x1, label='Regression for Validator 1', linestyle='--', color='#4E95D9')

    # Regression for x₂
    if np.any(valid_x2):
        reg_x2 = LinearRegression().fit(x_2[valid_x2].reshape(-1, 1), y[valid_x2])
        y_pred_x2 = reg_x2.predict(x_range)
        ax.plot(x_range, y_pred_x2, label='Regression for Validator 2', linestyle='--', color='#FF827B')

    # Axis labels with Futura
    ax.set_xlabel("Bullshit Score Assigned by the Validators", fontname='Futura', fontsize=16, labelpad=10)
    ax.set_ylabel("Bullshit Score Assigned by the LLM", fontname='Futura', fontsize=16, labelpad=10)

    # Make sure tick labels use Futura as well
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Futura')
        label.set_fontsize(12)

    # Custom legend entries for the data points, added last
    custom_legend = [
        Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=8, label='Data Point of Validator 1'),
        Line2D([0], [0], marker='o', color='black', linestyle='None', markersize=8, markerfacecolor='none', label='Data Point of Validator 2')
    ]

    ax.legend(handles=ax.get_legend_handles_labels()[0] + custom_legend, fontsize=12)
    plt.tight_layout()

    # Save the figure
    plt.savefig("figures/instrument_validation/uniform_instrument_validation.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_word_cloud(words):
    """
    Plots the word cloud for bureaucratic words.

    Parameters:
    -----------
    words: pd.DataFrame
        A DataFrame with columns 'word' and 'count'.

    Returns:
    --------
    None
    """
    # Filter out rare words
    words = words[words["count"] > 1].copy()
    words = words.sort_values("count", ascending=False)

    # Step 1: Get unique counts, sorted ascending
    unique_counts = sorted(words["count"].unique())

    # Step 2: Build a mapping from count → ordinal rank
    ordinal_map = {count: rank + 1 for rank, count in enumerate(unique_counts)}

    # Step 3: Apply the mapping to the "count" column
    words["ordinal"] = words["count"].map(ordinal_map)

    # Create dictionary of word frequencies
    word_freq = dict(zip(words["word"], words["count"]))
    word_ordinal = dict(zip(words["word"], words["ordinal"]))

    # Normalize counts for color mapping
    norm = Normalize(vmin=words["ordinal"].min(), vmax=words["ordinal"].max())
    # cmap = sns.color_palette("magma", as_cmap=True)

    # Define your custom color list
    my_colors = ["#7BBACE", "#367BCF", "#C0383D", "#EC7A3C"]

    # Create the custom colormap
    cmap = LinearSegmentedColormap.from_list("custom_cmap", my_colors)

    # Make my own cmap


    # Map each word to a normalized color based on count
    count_map = {word: cmap(norm(count)) for word, count in word_ordinal.items()}

    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        rgba = count_map.get(word, (0, 0, 0))  # default black if word not found
        r, g, b, _ = [int(255 * c) for c in rgba]
        return f"rgb({r}, {g}, {b})"

    # Create word cloud
    wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        color_func=color_func,
        prefer_horizontal=1.0,
        font_path="/System/Library/Fonts/Supplemental/Futura.ttc",  # Adjust if needed
        max_font_size=150,
        min_font_size=10,
        random_state=38,
    ).generate_from_frequencies(word_freq)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("figures/instrument_validation/bureaucratic_words.png", dpi=300, bbox_inches='tight')



def plot_logit_analysis(df):
    """
    Runs a logistic regression and plots the predicted probabilities.

    Parameters:
    -----------
    df: pd.DataFrame
        Must contain two columns:
        - 'bs_score_binary_dict': binary outcome (0 = not bullshit, 1 = bullshit)
        - 'bs_score_validated_random': continuous predictor

    Returns:
    --------
    None
    """
    # Drop missing values (if any)
    df = df.dropna(subset=["bs_score_binary_dict", "bs_score_validated_random"])

    # 1. Run logistic regression
    X = sm.add_constant(df["bs_score_validated_random"])
    y = df["bs_score_binary_dict"]

    model = sm.Logit(y, X)
    result = model.fit(disp=False)
    
    print(result.summary())

    # 2. Plot predicted probabilities
    x_vals = np.linspace(df["bs_score_validated_random"].min(), df["bs_score_validated_random"].max(), 300)
    x_vals_const = sm.add_constant(x_vals)
    pred_probs = result.predict(x_vals_const)

    plt.figure(figsize=(10, 6))
    
    # Scatter of actual data
    sns.stripplot(data=df, x="bs_score_validated_random", y="bs_score_binary_dict", jitter=0.1, alpha=0.4)

    # Line of predicted probabilities
    plt.plot(x_vals, pred_probs, color="black", linewidth=2, label="Logit prediction")

    plt.xlabel("Validated Random Score")
    plt.ylabel("Probability of 'Bullshit'")
    plt.title("Logistic Regression: Probability of Bullshit Assessment")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_ratings_regressions(df) -> None:
    # Filter all null values
    df = df[df["bs_score_llm"].notnull() & df["bs_score_binary_dict"].notnull() & df["rating"].notnull()]

    # Set the font to Futura
    mpl.rcParams['font.family'] = 'Futura'

    plt.figure(figsize=(8, 6))

    # Count duplicates and normalize sizes
    def get_sizes(x, y):
        counts = Counter(zip(x, y))
        sizes = [counts[(a, b)] for a, b in zip(x, y)]
        # Normalize to range 20–200
        sizes = np.interp(sizes, (min(sizes), max(sizes)), (20, 200))
        return sizes


    # Regression lines
    sns.regplot(x=df["bs_score_binary_dict"], y=df["rating"], scatter=False, color="#FF827B", label="Dict. BS-Score Regression", line_kws={"linestyle": "--"})
    sns.regplot(x=df["bs_score_llm"], y=df["rating"], scatter=False, color="#4E95D9", label="LLM BS-Score Regression", line_kws={"linestyle": "--"})

    # Plot Dictionary scores
    sizes_dict = get_sizes(df["bs_score_binary_dict"], df["rating"])
    plt.scatter(df["bs_score_binary_dict"], df["rating"], 
                s=sizes_dict, alpha=1.0, label="Data Point of Dict. BS-Score", marker='x', color="#a0a0a0", linewidths=0.5)
    
    # Plot LLM scores
    sizes_llm = get_sizes(df["bs_score_llm"], df["rating"])
    plt.scatter(df["bs_score_llm"], df["rating"], 
                s=sizes_llm, alpha=1.0, label="Data Point of LLM BS-Score", marker='o', color="none", facecolors='none', edgecolors="#a0a0a0", linewidths=0.5)


    # Make the axis labels larger
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14, ticks=[1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])


    plt.xlabel("Bullshit Score", fontsize=16, labelpad=10)
    plt.ylabel("Rating (Stars)", fontsize=16, labelpad=10)
    plt.grid(False)
    plt.legend(loc="lower right", fontsize=14)
    plt.tight_layout()

    # Save the figure
    plt.savefig("figures/instrument_validation/ratings_regressions.png", dpi=300, bbox_inches='tight')

    plt.show()



if __name__ == "__main__":
    # Load the data
    df = pd.read_pickle("data/master_data.pkl")
    # df1 = df[df["bs_score_validated_uniform"].notnull()]
    # df2 = df[df["bs_score_validated_random"].notnull()]

    # # Define the bs_score_binary_dict and bs_score_llm columns
    # bs_score_binary_dict = df2[["bs_score_binary_dict", "bs_score_validated_random"]]
    # bs_score_llm = df1["bs_score_llm"]
    # x_1 = df1["bs_score_validator_1"] # colour = #95C591
    # x_2 = df1["bs_score_validator_2"] # colour = #2B3674
    # x = df1["bs_score_validated_uniform"] # Joint colour: #36868D
    
    # # Plot the uniform instrument validation
    # plot_simple_uniform_instrument_validation(bs_score_llm, x, x_1, x_2)

    # # Plot the word cloud
    # words = pd.read_pickle("data/bureaucratic_words.pkl")
    # plot_word_cloud(words)

    # # Plot the binary assessment
    # plot_logit_analysis(bs_score_binary_dict)

    # Plot the ratings regressions
    plot_ratings_regressions(df)