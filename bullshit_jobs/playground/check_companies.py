import pandas as pd


def check_time_and_company(df: pd.DataFrame) -> pd.DataFrame:
    """
    Check if the time and company columns are valid.
    """
    # Print the first 5 rows of the dataframe
    print(df)

    # Print the nr of unique company names
    print(f"Number of unique company names: {df['firm'].nunique()}")

    # Print all firm names
    print("All firm names:")
    print(df['firm'].unique())

    # Show the return_to_office_week_numeric by firm
    print("Return to office week numeric by firm:")
    print(df.groupby('firm')['return_to_office_week_numeric'].unique())

    # Show the return_to_office_week_numeric by firm
    print("Return to office week numeric by firm:")
    print(df.groupby('firm')['return_to_office_week_numeric'].unique())
    return df


def check_times(df):
    # Find the earliest and latest time in the dataframe
    earliest_time = df['date'].min()
    latest_time = df['date'].max()

    # Show the date_week_numeric min and max
    print("Earliest and latest time:")
    print(f"Date week numeric min: {df['date_week_numeric'].min()}")
    print(f"Date week numeric max: {df['date_week_numeric'].max()}")

    print(f"Earliest time: {earliest_time}")
    print(f"Latest time: {latest_time}")



if __name__ == "__main__":
    # Read the data
    # df = pd.read_csv("data/master_data_firm.csv")
    df = pd.read_csv("data/master_data_cross_section.csv")

    # Check the time and company columns
    # df = check_time_and_company(df)
    # Check the times
    check_times(df)
