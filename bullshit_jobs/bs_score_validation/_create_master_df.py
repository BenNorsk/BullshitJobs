from bullshit_jobs.load_data._load_data import _quick_load
import pandas as pd
import numpy as np


def _create_master_df() -> pd.DataFrame:
    """
    Creates a master DataFrame by merging the BS score DataFrame with the original DataFrame.

    Returns:
    --------
    pd.DataFrame
        The merged DataFrame.
    """
    # Load the BS score DataFrame
    df_bs_score = _quick_load("data_with_bs_scores.pkl")

    # Load the validation DataFrames
    df_bs_score_validator_1 = _quick_load("manual_evaluation/filled_in_excels/manual_evaluation_task_ibrahim_filled_in.xlsx", "xlsx")
    df_bs_score_validator_2 = _quick_load("manual_evaluation/filled_in_excels/manual_evaluation_task_maike_filled_in.xlsx", "xlsx")
    df_bs_score_validator_3 = _quick_load("manual_evaluation/filled_in_excels/manual_evaluation_task_charlotte_filled_in.xlsx", "xlsx")

    # Merge and rename the columns
    df_bs_score = df_bs_score.merge(df_bs_score_validator_1, on="review_id", how="left").drop(columns=["cons_y"]).rename(columns={"cons_x": "cons", "score_ibrahim": "bs_score_validator_1"})
    df_bs_score = df_bs_score.merge(df_bs_score_validator_2, on="review_id", how="left").drop(columns=["cons_y"]).rename(columns={"cons_x": "cons", "score_maike": "bs_score_validator_2"})
    df_bs_score = df_bs_score.merge(df_bs_score_validator_3, on="review_id", how="left").drop(columns=["cons_y"]).rename(columns={"cons_x": "cons", "bs_score_savannah": "bs_score_validator_3"})

    # Make a new column bs_score_validated_uniform
    df_bs_score["bs_score_validated_uniform"] = np.nan
    df_bs_score.loc[df_bs_score["bs_score_validator_1"].notnull(), "bs_score_validated_uniform"] = df_bs_score["bs_score_validator_1"]
    df_bs_score.loc[df_bs_score["bs_score_validator_2"].notnull(), "bs_score_validated_uniform"] = df_bs_score["bs_score_validator_2"]

    # Make a new column bs_score_validated_random
    df_bs_score["bs_score_validated_random"] = np.nan
    df_bs_score.loc[df_bs_score["bs_score_validator_3"].notnull(), "bs_score_validated_random"] = df_bs_score["bs_score_validator_3"]

    print("The BS score DataFrame has been merged with the validation DataFrames.")
    print(df_bs_score[(df_bs_score["bs_score_validated_uniform"].notnull()) | (df_bs_score["bs_score_validated_random"].notnull())])
    
    # Remove these two columns: bs_score_cont_dict,bs_score_cont_dict_norminvgauss,
    df_bs_score = df_bs_score.drop(columns=["bs_score_cont_dict", "bs_score_cont_dict_norminvgauss"])
    
    # Save the master DataFrame
    df_bs_score.to_pickle("data/master_data.pkl")
    df_bs_score.to_csv("data/master_data.csv", index=False)
    
    return df_bs_score


if __name__ == "__main__":
    # Create the master DataFrame
    df_master = _create_master_df()
