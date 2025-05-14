import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def simple_stats(df):
    # Print the number of unique firms
    print(f"The number of unique firms is: {len(df['firm'].unique())}")

    # Print all unique firms
    print(f"The unique firms are: {df['firm'].unique()}")

    # Print the number of unique sectors
    print(f"The number of unique sectors is: {len(df['sector'].unique())}")
    # Print all unique sectors
    print(f"The unique sectors are: {df['sector'].unique()}")


def timeline_plot():
    # Read in the master data
    df = pd.read_csv("data/master_data.csv")
    print(df)

    # Read in the original data
    df_orig = pd.read_csv("data/reviews.csv")
    print(df_orig)

    # Convert the date columns to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df_orig["date"] = pd.to_datetime(df_orig["date"], errors="coerce")

    # Get the min and max dates
    min_date = df["date"].min()
    max_date = df["date"].max()
    min_date_orig = df_orig["date"].min()
    max_date_orig = df_orig["date"].max()

    # Print the min and max dates
    print(f"The min date is: {min_date}")
    print(f"The max date is: {max_date}")
    print(f"The min date in the original data is: {min_date_orig}")
    print(f"The max date in the original data is: {max_date_orig}")

    # Create a plot of the count of monthly reviews
    df["month"] = df["date"].dt.to_period("M")
    df_orig["month"] = df_orig["date"].dt.to_period("M")

    # Count the number of reviews per month
    df_count = df.groupby("month").size().reset_index(name="count")
    df_orig_count = df_orig.groupby("month").size().reset_index(name="count")
    print(df_count)
    print(df_orig_count)

    # Convert Period[M] to Timestamp
    df_count["month"] = df_count["month"].dt.to_timestamp()
    df_orig_count["month"] = df_orig_count["month"].dt.to_timestamp()

    # Sort for plotting
    df_count = df_count.sort_values("month")
    df_orig_count = df_orig_count.sort_values("month")


    # Split df_orig_count into two parts: inside and outside the date window
    df_orig_inside = df_orig_count[(df_orig_count["month"] >= min_date) & (df_orig_count["month"] <= max_date)]
    df_orig_outside = df_orig_count[(df_orig_count["month"] < min_date) | (df_orig_count["month"] > max_date)]

    # Cut the outside data at the max-date
    df_orig_outside = df_orig_outside[df_orig_outside["month"] <= max_date]

    plt.figure(figsize=(8, 6))
    plt.rcParams["font.family"] = "Futura"

    # Plot master data
    sns.lineplot(data=df_count, x="month", y="count", label="Selected Sample", color="#4E95D9")

    # Plot original data inside range in red
    sns.lineplot(data=df_orig_inside, x="month", y="count", label="Original Data", color="#FF827B")

    # Plot original data outside range in grey (no label to avoid clutter)
    sns.lineplot(data=df_orig_outside, x="month", y="count", color="#efcbc9", linestyle="--")

    # Add min and max date lines
    plt.axvline(x=min_date, color="#a0a0a0", linestyle="--")
    plt.axvline(x=max_date, color="#a0a0a0", linestyle="--")

    # Axes labels and scale
    plt.xticks(rotation=45, fontsize=16)
    plt.yticks(fontsize=16)
    plt.ylabel("Nr. of Monthly Reviews", fontsize=20, labelpad=10)
    plt.xlabel("Year", fontsize=20, labelpad=10)
    plt.yscale("log")

    # Title and legend
    plt.legend(loc="upper left", fontsize=18)

    # Layout and save
    plt.tight_layout()
    plt.savefig("figures/timeline_plot.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_random_samples():
    # Set a seed of 42 for reproducibility
    np.random.seed(42)

    # Read in the master data
    df = pd.read_csv("data/master_data.csv")

    # Get a random sample of 10 firms
    random_reviews = np.random.choice(df["review_id"], size=5, replace=False)

    # Get the reviews for the random firms
    random_df = df[df["review_id"].isin(random_reviews)]

    # Columns
    cols = ["firm", "rating", "cons", "date", "bs_score_binary_dict", "bs_score_llm"]
    random_df = random_df[cols]

    print(random_df)

    # Write the random sample to a latex file
    with open("tables/random_sample.tex", "w") as f:
        f.write(random_df.to_latex(index=False, escape=False, float_format="%.1f"))


def count_of_included_companies():
    # Read in the master data
    df = pd.read_csv("data/master_data.csv")

    # Basic stats
    num_firms = df["firm"].nunique()
    num_sectors = df["sector"].nunique()
    print(f"The number of unique firms is: {num_firms}")
    print(f"The number of unique sectors is: {num_sectors}")

    # Count of observations per firm and sector
    firm_counts = df.groupby(["sector", "firm"]).size().reset_index(name="count")

    # Desired sector order and color mapping
    sector_order = ["technology", "manufacturing", "finance", "telecom"]
    color_map = {
        "technology": "#4E95D9",    # Blue
        "manufacturing": "#0BE9A5", # Green
        "finance": "#efcd0b",       # Yellow
        "telecom": "#FF827B"        # Red
    }

    # Filter and order
    firm_counts = firm_counts[firm_counts["sector"].isin(sector_order)]
    firm_counts["sector"] = pd.Categorical(firm_counts["sector"], categories=sector_order, ordered=True)
    firm_counts = firm_counts.sort_values(["sector", "count"], ascending=[True, False])

    # Set Futura font
    plt.rcParams["font.family"] = "Futura"

    # --- Custom grouped bar plot using matplotlib ---
    plt.figure(figsize=(8, 6))
    sectors = firm_counts["sector"].unique()

    # Determine positions
    group_spacing = 2
    bar_width = 0.35
    positions = []
    xtick_labels = []
    colors = []
    counts = []
    firm_labels = []
    current_x = 0
    bar_positions = []

    for sector in sector_order:
        sector_data = firm_counts[firm_counts["sector"] == sector]
        n = len(sector_data)
        sector_positions = [current_x + i * bar_width for i in range(n)]

        # Store center of group for tick
        center_pos = np.mean(sector_positions)
        positions.append(center_pos)
        xtick_labels.append(sector.capitalize())

        # Store firm-specific data
        bar_positions.extend(sector_positions)
        counts.extend(sector_data["count"])
        colors.extend([color_map[sector]] * n)
        firm_labels.extend(sector_data["firm"])

        current_x = sector_positions[-1] + group_spacing

    # Plot bars with black border
    # bars = plt.bar(bar_positions, counts, width=bar_width, color=colors, edgecolor="black")
    # Plot bars with colored borders and semi-transparent fill
    bars = []
    for x, y, color in zip(bar_positions, counts, colors):
        bar = plt.bar(
            x,
            y,
            width=bar_width,
            color="#e0e0e0",           # fill color with alpha
            edgecolor="#a0a0a0",       # border color (fully opaque)
            linewidth=1,
            alpha=0.6              # semi-transparent fill
        )
        bars.append(bar[0])  # bar() returns a container


    # Label top and bottom firm per sector
    for sector in sector_order:
        sector_data = firm_counts[firm_counts["sector"] == sector]
        if sector_data.empty:
            continue
        # top_firm = sector_data.iloc[0]
        # bottom_firm = sector_data.iloc[-1]

        # label_offset = 3  # raise labels a bit more above the bar
        # line_extension = 1.5  # height of the vertical line

        # for firm_row in [top_firm, bottom_firm]:
        #     idx = firm_counts.index.get_loc(firm_row.name)
        #     bar = bars[idx]
        #     height = bar.get_height()

        #     # Draw vertical line from top of bar to just below label
        #     plt.plot(
        #         [bar.get_x() + bar.get_width() / 2] * 2,
        #         [height, height + label_offset - 0.5],  # line height stops below text
        #         color="black",
        #         linestyle="dotted",
        #         linewidth=1,
        #         zorder=1000
        #     )

        #     # Add label
        #     plt.text(
        #         bar.get_x(),                        # Left align with bar
        #         height + label_offset,             # Slightly higher
        #         firm_row["firm"],
        #         ha="left",
        #         va="bottom",
        #         fontsize=12,
        #         rotation=45,
        #         fontweight="bold"
        #     )



    # Axis and layout
    plt.xticks(positions, xtick_labels, fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("Sector", fontsize=20, labelpad=10)
    plt.ylabel("Number of Reviews", fontsize=20, labelpad=10)
    # Adding padding
    plt.margins(x=0.1, y=0.1)  # 5% padding on x-axis, 10% on y-axis

    plt.tight_layout()
    plt.savefig("figures/grouped_barplot_cleaned.png", dpi=300)
    plt.show()


def summary_stats():
    # Read in the master data
    df_master = pd.read_csv("data/master_data.csv")

    # Select the variables of interest
    cols = ["firm", "sector"]

    # Create a new DataFrame with the selected columns
    df = df_master[cols]

    # Give summary statistics on the unique, mode, and anti-mode values
    # Get the unique values
    unique_values = df.nunique()
    # Get the mode values
    mode_values = df.mode().iloc[0]
    # Get the anti-mode values
    anti_mode_values = df.apply(lambda x: x.value_counts().idxmin())
    # Combine the results into a new DataFrame
    summary_df = pd.DataFrame({
        "Unique": unique_values,
        "Mode": mode_values,
        "Anti-Mode": anti_mode_values
    })
    # Write the summary to a latex file
    print(summary_df)

    # Give the mean, std. dev., min, and max values of the rating
    # Get the mean values
    df_rating = df_master[["rating"]]
    mean_values = df_rating.mean()
    # Get the std. dev. values
    std_values = df_rating.std()
    # Get the min values
    min_values = df_rating.min()
    # Get the max values
    max_values = df_rating.max()
    # Combine the results into a new DataFrame
    summary_rating_df = pd.DataFrame({
        "Mean": mean_values,
        "Std. Dev.": std_values,
        "Min": min_values,
        "Max": max_values
    })
    print(summary_rating_df)

def show_min_max_dates():
    # Read in the master data
    df = pd.read_csv("data/master_data.csv")

    # Convert the date columns to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Get the min and max dates
    min_date = df["date"].min()
    max_date = df["date"].max()

    # Print the min and max dates
    print(f"The min date is: {min_date}")
    print(f"The max date is: {max_date}")

if __name__ == "__main__":
    # Load the data
    # df = pd.read_csv("data/master_data.csv")

    # simple_stats(df)
    # timeline_plot()
    # plot_random_samples()
    # count_of_included_companies()
    # summary_stats()
    show_min_max_dates()