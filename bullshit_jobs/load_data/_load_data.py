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
    df = df.drop_duplicates()

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
    path = pathlib.Path(__file__).parent.parent.parent / "data" / "reviews.csv"
    df = pd.read_csv(path)

    # Load in the selected companies
    excel = pathlib.Path(__file__).parent.parent.parent / "data" / "Tracking_Return_To_Office.xlsx"
    sheet = "RTO"
    selected_companies = pd.read_excel(excel, sheet_name=sheet)

    # Clean the data
    df = _clean_data(df, selected_companies)
    print("The data has been loaded and cleaned:")
    print(df)
    
    return df


def _save_data(df: pd.DataFrame, filename: str = "data") -> None:
    """
    Saves the data to a pkl file.

    Parameters:
    -----------
    filename: str
        The name of the file to save the data to.
    """
    path = pathlib.Path(__file__).parent.parent.parent / "data" / filename
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
    _save_data(df)
    return df


def _quick_load(filename: str = "data.pkl") -> pd.DataFrame:
    """
    Loads a saved dataframe from the data folder.

    Parameters:
    -----------
    filename: str
        The name of the file to load (pkl).

    Returns:
    --------
    df: pd.DataFrame
        The loaded DataFrame.
    """
    path = pathlib.Path(__file__).parent.parent.parent / "data" / filename
    df = pd.read_pickle(path)
    print(f"The data has been loaded from {path}")
    return df




