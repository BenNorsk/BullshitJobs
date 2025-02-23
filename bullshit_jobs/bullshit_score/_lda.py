import pandas as pd
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from bullshit_jobs.load_data import _load_data

def _create_lda_model(df: pd.DataFrame, col: str = "cons_processed", topics: int = 10) -> LdaModel:
    """
    Create an LDA model treating each row in df[col] as a document.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        col (str): Column name containing preprocessed text.
        topics (int): Number of topics for LDA.
    
    Returns:
        LdaModel: Trained LDA model.
    """
    # Ensure the column exists
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Convert each row into a list of words (assuming space-separated tokens)
    doc_clean = [row.split() for row in df[col]]
    
    # Create dictionary
    dictionary = Dictionary(doc_clean)
    dictionary.filter_extremes(no_below=10, no_above=0.33, keep_n=1000)
    
    # Create document-term matrix
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]
    
    # Train LDA model
    lda_model = LdaModel(doc_term_matrix, num_topics=topics, id2word=dictionary, passes=3)
    
    return lda_model


# Print the resulting topics
df = _load_data._quick_load("data_processed.pkl")
lda_model = _create_lda_model(df = df, col="cons_processed", topics=20)
for idx, topic in lda_model.show_topics(formatted=True, num_topics=20, num_words=15):
    print(f"Topic {idx}: {topic}")


# Get the topic distribution for each document
topic_distribution = lda_model.transform(df["cons_processed"])

# Get the score for Topic 8 (index 7, as topics are 0-indexed)
topic_8_scores = topic_distribution[:, 8]

# Add these scores as a new column to the dataframe
df['topic_8_score'] = topic_8_scores

# Sort the dataframe by the topic 8 score in descending order and get the top 10
top_10_topic_8 = df.sort_values(by="topic_8_score", ascending=False).head(10)

# Print the top 10 rows
print(top_10_topic_8["cons_processed"])

