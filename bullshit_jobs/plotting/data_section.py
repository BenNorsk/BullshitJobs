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
    plt.legend(loc="upper left", fontsize=14)

    # Layout and save
    plt.tight_layout()
    plt.savefig("figures/timeline_plot.png", dpi=300, bbox_inches='tight')
    plt.show()




if __name__ == "__main__":
    # Load the data
    # df = pd.read_csv("data/master_data.csv")

    # simple_stats(df)
    timeline_plot()