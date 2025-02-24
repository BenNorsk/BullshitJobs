from bullshit_jobs.load_data._load_data import _quick_load
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np



def _validate_bs_score(
        df: pd.DataFrame,
        bs_score_col: str,
        comparison_col: str = "rating"
    ) -> pd.DataFrame:
    """
    Validates the bullshit score by comparing it to the continuous dictionary bullshit score.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame to validate.
    bs_score_col: str
        The column containing the bullshit score.
    comparison_col: str
        The column to compare the bullshit score to.

    Returns:
    --------
    pd.DataFrame
        The filtered DataFrame containing only rows with valid scores.
    """
    # Control variables
    control_variables = ["Career Opportunities", "Compensation and Benefits", "Senior Management", "Work/Life Balance", "Culture & Values", "Diversity & Inclusion"]

    filtered_vars = [bs_score_col, comparison_col, "firm"] + control_variables 
    # Filter rows where the bullshit score and rating are not null
    df_filtered = df.dropna(subset=filtered_vars)

    # Create dummy variables for firms (drop_first=True to avoid multicollinearity)
    firm_dummies = pd.get_dummies(df_filtered["firm"], prefix="firm", drop_first=True)

    # Further control variables
    control_variables.append(bs_score_col)

    # Combine the bs_score with firm dummies
    X = pd.concat([df_filtered[control_variables], firm_dummies], axis=1)
    X = X.astype(float)
    y = df_filtered[comparison_col].astype(float)

    # Fit the linear regression model
    X = sm.add_constant(X)
    model = sm.OLS(y, X)
    results = model.fit()

    # Print a model summary
    print(results.summary())

    return df_filtered


if __name__ == "__main__":
    # Load the data
    df = _quick_load("data_with_bs_score.pkl")

    # Validate the bullshit score
    _validate_bs_score(df, "bs_score_binary_dict")

    # Conclusion
    print("Only the binary dictionary-based bullshit score is validated.")
