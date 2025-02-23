import pandas as pd
from bullshit_jobs.load_data._load_data import _quick_load
from bullshit_jobs.load_data._load_data import _save_data
from bullshit_jobs.load_data._load_data import _load_new_and_save
from bullshit_jobs.preprocessing.preprocessing import _preprocess_column



def _create_list_of_bureaucratic_words():
    """
    Create a list of bureaucratic words from the raw data.

    Returns:
    --------
    df: pd.DataFrame
        The DataFrame containing the list of bureaucratic words and the number of times each word appears.
    """
    
    # Read in the list of bureaucratic words
    filename = "bureaucratic_words_raw.csv"

    # Load the words
    words = _quick_load(filename, filetype="csv")

    # Pre-process the words
    df = _preprocess_column(words, col="word")

    # Count the number of times each word appears
    word_processed = [word for phrase in df["word_processed"] for word in phrase.split()]
    df_word_processed = pd.DataFrame(word_processed, columns=["word_processed"])
    word_counts = df_word_processed["word_processed"].value_counts()

    # Make a df of the word counts with a column "word" and a column "count"
    df = pd.DataFrame(word_counts)
    df.reset_index(inplace=True)
    df.columns = ["word", "count"]

    # Save the processed list of bureaucratic words
    _save_data(df, filename="bureaucratic_words")

    # Print all words which occur more than once
    return df



if __name__ == "__main__":
    _create_list_of_bureaucratic_words()
