import pandas as pd
from bullshit_jobs.load_data._load_data import _quick_load, _save_data
import pandas as pd
import math
from scipy.stats import norminvgauss


def _assign_continuous_bs_score(
        text: str,
        words: pd.DataFrame,
        threshold: int = 1
    ) -> int:
    """
    Assigns a continuous bullshit score to a text based on the presence of certain (bullshit) words.

    Parameters:
    -----------
    text: str
        The text to assign the score to.
    words: pd.DataFrame
        The DataFrame containing the bullshit words and their counts.
    threshold: int
        The minimum count for a word to be considered.

    Returns:
    --------
    bs_score: int
        The continuous bullshit score.
    """
    # BS Score
    bs_score = 0

    # Split the text into a list of words
    text = text.split()

    # Create the word list
    word_list = words[words["count"] >= threshold]["word"].tolist()

    # Add the score for each word in the text
    for word in text:
        if word in word_list:
            score_addition = words[words["word"] == word]["count"].values[0]
            bs_score += score_addition
    
    return bs_score


def _make_continuous_dict_bs_score(
        df: pd.DataFrame,
        words: pd.DataFrame,
        input_col: str = "cons_processed",
        output_col: str = "bs_score_cont_dict",
        threshold: int = 1
    ) -> pd.DataFrame:
    
    # Apply the _assign_continuous_bs_score function to the DataFrame
    df[output_col] = df[input_col].apply(lambda x: _assign_continuous_bs_score(x, words, threshold))

    # Normalise the BS score by dividing the output_col by the length of the input_col.split()
    df[output_col] = (df[output_col] / df[input_col].apply(lambda x: len(x.split()))) * math.log(1 + len(df[input_col].apply(lambda x: len(x.split()))))

    # Fit a normal inverse Gaussian distribution
    gaussian = df.copy()
    gaussian = gaussian[gaussian[output_col] > 0]
    params = norminvgauss.fit(gaussian[output_col])

    # Apply the normal inverse Gaussian distribution to the data
    df[f'{output_col}_norminvgauss'] = df[output_col].apply(lambda x: norminvgauss.cdf(x, params[0], params[1], params[2], params[3]) if x > 0 else 0)

    return df


def _make_binary_dict_bs_score(
        df: pd.DataFrame,
        words: pd.DataFrame,
        input_col: str = "cons_processed",
        output_col: str = "bs_score_binary_dict",
        threshold: int = 1
    ) -> pd.DataFrame:
    """
    Transform the text data into a binary score based on the presence of certain (bullshit) words.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the text data.
    words: list
        List of words to look for in the text.
    input_col: str
        The column containing the preprocessed text.
    output_col: str
        The column to save the binary score to.

    Returns:
    --------
    df: pd.DataFrame
        The DataFrame with the binary score
    """
    words = words[words["count"] >= threshold]["word"].tolist()
    df[output_col] = df[input_col].apply(lambda x: sum(word in x for word in words))
    df[output_col] = df[output_col].apply(lambda x: 1 if x else 0)
    return df


def _create_dictionary_bs_score() -> None:
    """
    Create a dictionary-based bullshit score for the text data.
    """

    # Load the data
    df = _quick_load("data_processed.pkl")

    # Load the list of bureaucratic words
    words = _quick_load("bureaucratic_words.pkl")

    # Create the continuous dictionary-based bullshit score
    df = _make_continuous_dict_bs_score(df, words, threshold=0)

    # Create the binary dictionary-based bullshit score
    df = _make_binary_dict_bs_score(df, words)

    # Descibre bs_score_cont_dict
    print(df['bs_score_cont_dict'].describe())

    # Descibre bs_score_cont_dict_norminvgauss
    print(df['bs_score_cont_dict_norminvgauss'].describe())


    # Descibre bs_score_binary_dict
    print(df['bs_score_binary_dict'].describe())
    
    # Save the data
    try:
        df_with_bs_score = _quick_load("data_with_bs_score.pkl", filetype="pkl")
        print("The data is concatenated with the bs_score columns:")

        # Drop the columns if they already exist
        df_with_bs_score.drop(columns=['bs_score_cont_dict', 'bs_score_cont_dict_norminvgauss', 'bs_score_binary_dict'], inplace=True)

        # Merge the dataframes on the 'id' column and keep the id column
        df_with_bs_score = pd.merge(df_with_bs_score, df[['id', 'bs_score_cont_dict', 'bs_score_cont_dict_norminvgauss' 'bs_score_binary_dict']], on='id')


    except:
        print("The data with the bs_score does not exist yet")
        _save_data(df, filename="data_with_bs_score")


if __name__ == "__main__":
    _create_dictionary_bs_score()
