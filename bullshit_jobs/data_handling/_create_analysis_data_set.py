import pandas as pd
import datetime


def _create_analysis_data_set(
        df: pd.DataFrame,
        Y_col: str,
        firm: str,
        control_firms: list,
        date_col: str,
        treatment_date: datetime.datetime,
        data_start_date: datetime.datetime,
        data_end_date: datetime.datetime
    ) -> pd.DataFrame:
    """
    Create a data set ready for synthetic control analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The data set to be used.
    Y_col : str
        The column name of the outcome variable.
    firm : str
        The firm that is treated.
    control_firms : list
        A list of firms that are used as controls.
    date_col : str
        The column name of the date variable.
    treatment_date : datetime.datetime
        The date of the treatment.
    data_start_date : datetime.datetime
        The start date of the data set.
    data_end_date : datetime.datetime
        The end date of the data set.

    Returns
    -------
    pd.DataFrame
        A data set ready for synthetic control analysis.
    """
    print(df)
    # Select the relevant firms
    all_firms = control_firms + [firm]

    # Remove all firms that are not in the data set
    df = df[df['firm'].isin(all_firms)]

    # Remove all columns that are not needed
    df = df[[date_col, 'firm', Y_col]]

    # Trim the data set to the relevant time period
    df = df[(df[date_col] >= data_start_date) & (df[date_col] <= data_end_date)]

    # Create the treatment indicator (1 if date_col > treatment_date and firm is treated, 0 otherwise)
    df['treatment'] = 0
    df.loc[(df[date_col] >= treatment_date) & (df['firm'] == firm), 'treatment'] = 1


    # Aggregate the data per firm on a monthly basis, taking the mean of the outcome variable

    # Ensure 'date_col' is a datetime object
    df[date_col] = pd.to_datetime(df[date_col])

    # Aggregate the data per firm on a monthly basis, taking the mean of the outcome variable
    df = df.groupby(['firm', 'treatment', pd.Grouper(key=date_col, freq='M')])[Y_col].mean().reset_index()

    # Remove the day component, keeping only the year and month
    df[date_col] = df[date_col].dt.to_period('M')

    print(df)
    return df
