import pandas as pd
import pathlib
import re
import numpy as np


def _clean_data(df: pd.DataFrame, selected_companies: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the data by removing duplicates and missing values.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame to clean.
    
    selected_companies: pd.DataFrame
        A DataFrame containing the selected companies.

    Returns:
    --------
    df: pd.DataFrame
        The cleaned DataFrame.
    """
    # Create a new 'firm' field in the reviews
    re = r'^(.*Reviews\/)(.*)(\-Reviews)'
    df['firm'] = df['firm_link'].str.extract(re, expand=False)[1]

    # Keep only the selected companies
    df = df[df['firm'].isin(selected_companies['firm'])]

    # Add the field 'return_to_office' and 'sector from the selected companies to the reviews
    selected_companies = selected_companies[["firm", "return_to_office", "sector"]]
    df = df.merge(selected_companies, on="firm", how="left")

    # Remove leading/trailing spaces, replace empty strings with NaN
    df['date'] = df['date'].astype(str).str.strip().replace("", pd.NA)

    # Drop rows where 'date' is NaN
    df = df.dropna(subset=['date'])

    # Convert date column, allowing for mixed formats
    df['date'] = pd.to_datetime(df['date'], format='%b %d, %Y', errors='coerce')

    # Drop rows where conversion failed
    df = df.dropna(subset=['date'])

    # Dataframe Pre-cutting the date
    print("Dataframe before cutting the date:")
    print(df)

    # Filter dates within range
    df = df[(df['date'] < '2023-01-08') & (df['date'] > '2020-04-01')]

    # Convert the return_to_office column to datetime # 2022-06-01
    df['return_to_office'] = pd.to_datetime(df['return_to_office'], format='%Y-%m-%d', errors='coerce')

    # Remove duplicates
    df = df.drop_duplicates()

    # Create a new column 'id' with a unique identifier
    df['review_id'] = range(1, len(df) + 1)

    return df


def _load_data() -> pd.DataFrame:
    """
    Reads in the data from the reviews.csv file and the selected companies from the Tracking_Return_To_Office.xlsx file.

    Returns:
    --------
    df: pd.DataFrame
        A DataFrame containing the selected reviews.
    """

    # Load in the reviews
    path = f'./data/reviews.csv'
    df = pd.read_csv(path)

    # Load in the selected companies
    excel = f'data/Tracking_Return_To_Office.xlsx'
    sheet = "RTO"
    selected_companies = pd.read_excel(excel, sheet_name=sheet)

    # Clean the data
    print("The data is being loaded and cleaned...")
    print(df)
    df = _clean_data(df, selected_companies)
    print("The data has been loaded and cleaned:")

    # Show the first few rows
    print(df)
    
    return df


def _save_data(df: pd.DataFrame, filename: str) -> None:
    """
    Saves the data to a pkl file.

    Parameters:
    -----------
    filename: str
        The name of the file to save the data to.
    """
    path = f'./data/{filename}'
    df.to_pickle(f'{path}.pkl')
    df.to_csv(f'{path}.csv', index=False)

    print(f"The data has been saved as {path} (.csv and .pkl)")


def _load_new_and_save() -> pd.DataFrame:
    """
    Loads the data, preprocesses it and saves it.

    Returns:
    --------
    df: pd.DataFrame
        The preprocessed DataFrame.
    """
    df = _load_data()
    _save_data(df, "data")
    return df


def _quick_load(filename: str, filetype: str = "pkl") -> pd.DataFrame:
    """
    Loads a saved dataframe from the data folder.

    Parameters:
    -----------
    filename: str
        The name of the file to load (pkl).
    filetype: str
        The type of file to load (pkl, xlsx, or csv).

    Returns:
    --------
    df: pd.DataFrame
        The loaded DataFrame.
    """
    path = f'./data/{filename}'
    if filetype == "csv":
        df = pd.read_csv(path)
    elif filetype == "pkl":
        df = pd.read_pickle(path)
    elif filetype == "xlsx":
        df = pd.read_excel(path)
    else:
        raise ValueError("Filetype must be 'csv' or 'pkl' or 'xlsx'.")
    print(f"The data has been loaded from {path}")
    return df

def _remerge_data() -> pd.DataFrame:
    """
    Merges the data from the three dataframes.

    Returns:
    --------
    df: pd.DataFrame
        The merged DataFrame.
    """
    # Load the data
    df1 = _quick_load("llm/data_with_bs_score_1.pkl")
    df2 = _quick_load("llm/data_with_bs_score_2.pkl")
    df3 = _quick_load("llm/data_with_bs_score_3.pkl")
    
    # Merge the data
    df = pd.concat([df1, df2, df3])

    # Drop all columns where either bs_score_binary_dict or bs_score_llm is NaN
    df = df.dropna(subset=["bs_score_binary_dict", "bs_score_llm"])

    # Turn the bs_score_llm into a float
    df["bs_score_llm"] = df["bs_score_llm"].astype(float)

    # Replace 0.75 with 0.8 and 0.85 with 0.9
    df["bs_score_llm"] = df["bs_score_llm"].replace(0.75, 0.8).replace(0.85, 0.9)

    # Show the unique values of bs_score_llm
    print(df["bs_score_llm"].unique())

    # Show the summary statistics of bs_score_llm
    print(df["bs_score_llm"].describe())

    # Save the data
    _save_data(df, "data_with_bs_scores")

    return df


def add_week_numeric_columns(df, date_col='date', rto_col='return_to_office'):
    # Convert to datetime if needed
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df[rto_col] = pd.to_datetime(df[rto_col], errors='coerce')

    # Compute the reference week (week 0) from the date column
    min_week = df[date_col].dropna().min().to_period('W').start_time

    # Compute number of weeks since min_week, handling NAs properly
    df['date_week_numeric'] = df[date_col].apply(
        lambda d: ((d - min_week).days // 7) if pd.notna(d) else np.nan
    ).astype('Int64')  # allows for NA-safe integers

    df['return_to_office_week_numeric'] = df[rto_col].apply(
        lambda d: ((d - min_week).days // 7) if pd.notna(d) else np.nan
    ).astype('Int64')

    return df


def add_firm_id_to_master_data() -> pd.DataFrame:
    """
    Adds the firm_id to the master data.

    Returns:
    --------
    df: pd.DataFrame
        The DataFrame with the firm_id added.
    """
    # Load the data
    df = _quick_load("master_data.pkl")

    # Print the data
    print(df)

    # Get all unique firms
    firms = df["firm"].unique()
    print(f"The unique firms are: {firms}")

    # Print the number of unique firms
    print(f"The number of unique firms is: {len(firms)}")

    # Order the firms alphabetically
    firms = sorted(firms)

    # Create a dictionary with the firm_id
    firm_id_dict = {firm: i for i, firm in enumerate(firms)}
    print(f"The firm_id dictionary is: {firm_id_dict}")

    # Add a new firm_id column to the dataframe
    df["firm_id"] = df["firm"].map(firm_id_dict)
    print(f"The firm_id column has been added to the dataframe: {df}")

    print(df)

    # Transform the weeks
    df = add_week_numeric_columns(df, date_col='date', rto_col='return_to_office')
    print(f"The weeks have been transformed: {df}")

    print(df)

    # Aggregate the data on a firm level. Keep the columns firm_id, firm, date_week_numeric, return_to_office_week_numeric
    # bs_score_binary_dict (average), bs_score_llm (average), rating (average), review_id (count)
    # Career Opportunities	Compensation and Benefits	Senior Management	Work/Life Balance	Culture & Values	Diversity & Inclusion
    # List of columns to average
    score_columns = [
        'bs_score_binary_dict', 'bs_score_llm', 'rating',
        'Career Opportunities', 'Compensation and Benefits',
        'Senior Management', 'Work/Life Balance',
        'Culture & Values', 'Diversity & Inclusion'
    ]

    # Convert all score columns to numeric
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Perform aggregation
    agg_df = df.groupby(['firm_id', 'firm', 'date', 'date_week_numeric', 'return_to_office', 'return_to_office_week_numeric']).agg(
        review_count=('review_id', 'count'),
        **{col: (col, 'mean') for col in score_columns}
    ).reset_index()

    # Reorder the columns
    cols = score_columns + ['firm_id', 'firm', 'date', 'date_week_numeric', 'return_to_office', 'return_to_office_week_numeric'] + ['review_count']
    agg_df = agg_df[cols]

    # Print the unique return_to_office values
    print(f"The unique return_to_office values are: {agg_df['return_to_office'].unique()}")

    print(agg_df)

    # Save the data
    _save_data(agg_df, "master_data_panel")

    return df

def make_master_data_cross_section():
    # Load the data
    df = _quick_load("master_data.pkl")

    # Print the data
    print(df)

    # Get all unique firms
    firms = df["firm"].unique()
    print(f"The unique firms are: {firms}")

    # Print the number of unique firms
    print(f"The number of unique firms is: {len(firms)}")

    # Order the firms alphabetically
    firms = sorted(firms)

    # Create a dictionary with the firm_id
    firm_id_dict = {firm: i for i, firm in enumerate(firms)}
    print(f"The firm_id dictionary is: {firm_id_dict}")

    # Add a new firm_id column to the dataframe
    df["firm_id"] = df["firm"].map(firm_id_dict)
    print(f"The firm_id column has been added to the dataframe: {df}")

    print(df)

    # Transform the weeks
    df = add_week_numeric_columns(df, date_col='date', rto_col='return_to_office')
    print(f"The weeks have been transformed: {df}")

    print(df)
    print(df.columns)

    # If return_to_office_week_numeric is NaN, set it to 0
    df.loc[df["return_to_office_week_numeric"].isna(), "return_to_office_week_numeric"] = 0

    # One hot encode each sector
    sectors = df["sector"].unique()
    print(f"The unique sectors are: {sectors}")

    # Create a new column for each sector
    for sector in sectors:
        df[f'sector_{sector}'] = np.where(df["sector"] == sector, 1, 0)
    print(f"The sectors have been one hot encoded: {df}")
    print(df)

    # Save as a cross-section
    _save_data(df, "master_data_cross_section")


def change_master_card_to_finance():
    df = _quick_load("master_data.pkl")
    # Change the sector of "Mastercard" to "finance"
    df.loc[df["firm"] == "Mastercard", "sector"] = "finance"
    # Save the data
    _save_data(df, "master_data")
    return df


if __name__ == "__main__":
    # Load the data
    # df = _load_new_and_save()
    # print(df)
    # add_firm_id_to_master_data()
    # change_master_card_to_finance()
    make_master_data_cross_section()







