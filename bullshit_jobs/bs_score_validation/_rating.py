from bullshit_jobs.load_data._load_data import _quick_load
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import matplotlib.pyplot as plt



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
    """
    # Filter rows where the bullshit score is greater than 0
    df_filtered = df[df[bs_score_col] > 0]

    # Fit a linear regression model
    X = df_filtered[bs_score_col].values.reshape(-1, 1)
    y = df_filtered[comparison_col].values

    model = LinearRegression()
    model.fit(X, y)

    # Print summary statistics using statsmodels for more detailed output
    X_with_const = sm.add_constant(X)  # Add constant term (intercept)
    ols_model = sm.OLS(y, X_with_const)
    ols_results = ols_model.fit()
    print(ols_results.summary())

    # Predictions
    y_pred = model.predict(X)

    # Plot actual vs. predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(df_filtered[bs_score_col], y, label="Actual values", color="blue")
    plt.plot(df_filtered[bs_score_col], y_pred, color="red", label="OLS Regression Line")
    plt.xlabel(bs_score_col)
    plt.ylabel(comparison_col)
    plt.title(f"Actual vs Predicted: {bs_score_col} vs {comparison_col}")
    plt.legend()
    plt.show()

    return df_filtered


if __name__ == "__main__":
    # Load the data
    df = _quick_load("data_with_bs_score.pkl")

    # Validate the bullshit score
    _validate_bs_score(df, "bs_score_cont_dict")