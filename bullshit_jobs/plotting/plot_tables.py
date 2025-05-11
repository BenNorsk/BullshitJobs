import pandas as pd
import statsmodels.api as sm


def run_ols(x,y):
    """
    Runs an OLS regression and returns the fitted model.
    
    Parameters:
    -----------
    x: pd.Series
        The independent variable.
    y: pd.Series
        The dependent variable.

    Returns:
    --------
    sm.OLS
        The fitted OLS model.
    """
    # Drop NaN values
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    print(model.summary())
    return model
    

def extract_ols_results(model, var_names_latex):
    summary = {
        "Coef.": model.params,
        "Std. Err.": model.bse,
        "CI\_2.5": model.conf_int().iloc[:, 0],
        "CI\_97.5": model.conf_int().iloc[:, 1],
        "p\_value": model.pvalues,
    }
    df = pd.DataFrame(summary)
    df.index = var_names_latex
    return df.T  # Transpose so metrics are rows


def create_combined_results_table(model_beta, model_alpha, model_gamma):
    # Define LaTeX variable names
    beta_names = [r"$\hat{\beta_{0}}$", r"$\hat{\beta_{1}}$"]
    alpha_names = [r"$\hat{\alpha_{0}}$", r"$\hat{\alpha_{1}}$"]
    gamma_names = [r"$\hat{\gamma_{0}}$", r"$\hat{\gamma_{1}}$"]

    # Extract tables
    df_beta = extract_ols_results(model_beta, beta_names)
    df_alpha = extract_ols_results(model_alpha, alpha_names)
    df_gamma = extract_ols_results(model_gamma, gamma_names)

    # Combine
    df_combined = pd.concat([df_beta, df_alpha, df_gamma], axis=1)

    # Round all values to 3 decimal places
    df_combined = df_combined.round(3)

    # Transpose so coefficients become columns
    df_transposed = df_combined.T
    df_transposed.columns.name = None
    df_transposed.index.name = "Coefficient"

    # Convert index to a column, then reorient
    df_final = df_transposed.reset_index().T
    df_final.columns = df_final.iloc[0]
    df_final = df_final.drop(df_final.index[0])
    df_final = df_final.rename_axis("Metric").reset_index(drop=True)

    # Add a new column for the metrics
    metrics_column = ["Coefficient", "Std. Err.", r"$CI_{2.5\%}$", r"$CI_{2.5\%}$", "p-value"]
    df_final["Metric"] = metrics_column

    # Desired column order
    ordered_columns = ["Metric"] + beta_names + alpha_names + gamma_names
    df_final = df_final[ordered_columns]



    return df_final



def create_model_stats_table(model_beta, model_alpha, model_gamma):
    def extract_stats(model):
        return [
            int(model.nobs),
            model.rsquared,
            model.fvalue,
            model.f_pvalue
        ]

    # Extract and organize
    data = {
        "Joint Model": extract_stats(model_beta),
        "Val. 1 Model": extract_stats(model_alpha),
        "Val. 2 Model": extract_stats(model_gamma)
    }

    # Create DataFrame
    df_stats = pd.DataFrame(
        data,
        index=["$n$", "$R^2$", "F-statistic", "p-value"]
    )

    return df_stats.round(3)


def llm_results_table(df):
    # Define the dependent and independent variables
    y = df["bs_score_llm"]
    x = df["bs_score_validated_uniform"]

    # Drop where bs_score_validator_1 is NaN
    df_1 = df[df["bs_score_validator_1"].notnull()]
    x_1 = df_1["bs_score_validator_1"]
    y_1 = df_1["bs_score_llm"]

    # Drop where bs_score_validator_2 is NaN
    df_2 = df[df["bs_score_validator_2"].notnull()]
    x_2 = df_2["bs_score_validator_2"]
    y_2 = df_2["bs_score_llm"]

    # Run OLS regression for the main model
    model_beta = run_ols(x, y)

    # Run OLS regression for the additional models
    model_alpha = run_ols(x_1, y_1)
    model_gamma = run_ols(x_2, y_2)

    # Create combined results table
    df_combined = create_combined_results_table(model_beta, model_alpha, model_gamma)

    # Create model stats table
    df_stats = create_model_stats_table(model_beta, model_alpha, model_gamma)
    print(df_stats)

    # Save the table to a LaTeX file
    with open("tables/instrument_validation/llm_ols_results_table.tex", "w") as f:
        f.write(df_combined.to_latex(escape=False, index=False, float_format="%.3f"))
    
    # Save the table to a LateX file
    with open("tables/instrument_validation/llm_model_stats_table.tex", "w") as f:
        f.write(df_stats.to_latex(escape=False, index=False, float_format="%.3f"))

    print(df_combined)


def ratings_results_table(df):
    # Define the dependent and independent variables
    x = df["bs_score_llm"]
    y = df["rating"]

    # Run OLS regression for the main model
    model_beta = run_ols(x, y)

    print(model_beta.summary())

    # Extract the results in a table, with the coefficients in the first column,
    # and the metrics in the first row
    df_results = pd.DataFrame({
        "Coef.": model_beta.params,
        "Std. Err.": model_beta.bse,
        r"$CI_{2.5\%}$": model_beta.conf_int().iloc[:, 0],
        r"$CI_{97.5\%}$": model_beta.conf_int().iloc[:, 1],
        "p-value": model_beta.pvalues,
    })
    df_results.index = ["Intercept", "bs_score_llm"]

    # Round all values to 3 decimal places
    df_results = df_results.round(3)

    # Save the results to a LaTeX file
    with open("tables/instrument_validation/llm_model_ratings_ols_results_table.tex", "w") as f:
        f.write(df_results.to_latex(escape=False, index=True, float_format="%.3f"))
    print(df_results)

    # Now repeat the same thing for x = bs_score_binary_dict
    x = df["bs_score_binary_dict"]
    y = df["rating"]

    model_delta = run_ols(x, y)

    print(model_delta.summary())
    # Extract the results in a table, with the coefficients in the first column,
    # and the metrics in the first row

    df_results = pd.DataFrame({
        "Coef.": model_delta.params,
        "Std. Err.": model_delta.bse,
        r"$CI_{2.5\%}$": model_delta.conf_int().iloc[:, 0],
        r"$CI_{97.5\%}$": model_delta.conf_int().iloc[:, 1],
        "p-value": model_delta.pvalues,
    })

    df_results.index = ["Intercept", "bs_score_binary_dict"]
    # Round all values to 3 decimal places

    df_results = df_results.round(3)
    # Save the results to a LaTeX file
    with open("tables/instrument_validation/dict_model_ratings_ols_results_table.tex", "w") as f:
        f.write(df_results.to_latex(escape=False, index=True, float_format="%.3f"))

def validators_results_table(df):
    # Define the dependent and independent variables
    x = df["bs_score_llm"]
    y = df["bs_score_validated_uniform"]

    # Run OLS regression for the main model
    model_beta = run_ols(x, y)

    print(model_beta.summary())

    # Extract the results in a table, with the coefficients in the first column,
    # and the metrics in the first row
    df_results = pd.DataFrame({
        "Coef.": model_beta.params,
        "Std. Err.": model_beta.bse,
        r"$CI_{2.5\%}$": model_beta.conf_int().iloc[:, 0],
        r"$CI_{97.5\%}$": model_beta.conf_int().iloc[:, 1],
        "p-value": model_beta.pvalues,
    })
    df_results.index = ["Intercept", "bs_score_llm"]

    # Round all values to 3 decimal places
    df_results = df_results.round(3)

    # Save the results to a LaTeX file
    with open("tables/instrument_validation/llm_model_validators_results_table.tex", "w") as f:
        f.write(df_results.to_latex(escape=False, index=True, float_format="%.3f"))
    print(df_results)

    # Now repeat the same thing for x = bs_score_binary_dict
    x = df["bs_score_binary_dict"]
    y = df["bs_score_validated_uniform"]

    model_delta = run_ols(x, y)

    print(model_delta.summary())
    # Extract the results in a table, with the coefficients in the first column,
    # and the metrics in the first row

    df_results = pd.DataFrame({
        "Coef.": model_delta.params,
        "Std. Err.": model_delta.bse,
        r"$CI_{2.5\%}$": model_delta.conf_int().iloc[:, 0],
        r"$CI_{97.5\%}$": model_delta.conf_int().iloc[:, 1],
        "p-value": model_delta.pvalues,
    })

    df_results.index = ["Intercept", "bs_score_binary_dict"]
    # Round all values to 3 decimal places

    df_results = df_results.round(3)
    # Save the results to a LaTeX file
    with open("tables/instrument_validation/dict_model_validators_ols_results_table.tex", "w") as f:
        f.write(df_results.to_latex(escape=False, index=True, float_format="%.3f"))

if __name__ == "__main__":
    # Load the data
    df = pd.read_pickle("data/master_data.pkl")
    df_uniform = df[df["bs_score_validated_uniform"].notnull()]

    # Save the bs_score_llm ols table 
    # llm_results_table(df_uniform)
    ratings_results_table(df)
    validators_results_table(df_uniform)