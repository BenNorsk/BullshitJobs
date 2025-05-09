import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bullshit_jobs.load_data._load_data import _quick_load
from bullshit_jobs.load_data._load_data import _save_data 


def _preprocess_text(text: str) -> str:
    """
    Runs a series of preprocessing steps on a text.

    Parameters:
    -----------
    text: str
        The text to preprocess.

    Returns:
    --------
    text: str
        The preprocessed
    """
    # Convert all text to a string
    text = str(text)

    # Initialise the word stemmer and stopwords
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # 1. Convert all text to lowercase
    text = text.lower()

    # 2. Remove punctuations
    text = re.sub(r'[^\w\s]', '', text)

    # 3. Replace numbers with #
    text = re.sub(r'\d+', '#', text)

    # 4. Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])

    # 5. Stemming
    text = ' '.join([stemmer.stem(word) for word in text.split()])

    print(f'The data has been preprocessed: {text}')
    return text


def _preprocess_column(df: pd.DataFrame, col: str = "cons") -> pd.DataFrame:
    """
    Preprocess the text in a specified column of a DataFrame.

    Parameters:
    -----------
    df: pd.DataFrame
        The DataFrame containing the text to preprocess.

    col: str
        The column containing the text to preprocess.

    Returns:
    --------
    df: pd.DataFrame
        The DataFrame with the preprocessed text in a new column.
    """
    # Download the stopwords
    nltk.download('stopwords')
    
    # Apply the preprocessing function to the specified column and save the result in a new column
    df[f'{col}_processed'] = df[col].apply(_preprocess_text)
    
    return df

if __name__ == "__main__":
    
    # Load the data
    df = _quick_load("data.pkl")
    
    # Preprocess the text in the cons column
    df = _preprocess_column(df, col="cons")
    
    print("The text has been preprocessed:")
    print(df)

    # Save the preprocessed data
    _save_data(df, filename="data_processed")

