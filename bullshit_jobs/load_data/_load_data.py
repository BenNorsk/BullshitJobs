import pandas as pd
import pathlib
import re


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
        The type of file to load (pkl or csv).

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
    else:
        raise ValueError("Filetype must be 'csv' or 'pkl'")
    print(f"The data has been loaded from {path}")
    return df


if __name__ == "__main__":
    # Load the data
    df = _load_new_and_save()
    print(df)






