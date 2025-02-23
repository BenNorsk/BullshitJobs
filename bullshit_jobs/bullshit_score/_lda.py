from gensim.models.ldamodel import LdaModel
from bullshit_jobs.load_data import _load_data
from bullshit_jobs.preprocessing.preprocessing import _preprocess_column

# def _create_guided_lda(
#         df: pd.DataFrame,
#         col: str = "cons_processed",
#         topics: int = 10,
#         seed_words: list = []
#     ) -> glda.GuidedLDA:

#     """
#     Create a guided LDA model with a given set of words assigned to one topic.

#     Parameters:
#         df (pd.DataFrame): DataFrame containing the text data.
#         col (str): Column name containing preprocessed text.
#         topics (int): Number of topics for LDA.
#         seed_words (list): List of words that should be associated with a specific topic.

#     Returns:
#         guidedlda.GuidedLDA: Trained guided LDA model.
#     """
#     if col not in df.columns:
#         raise ValueError(f"Column '{col}' not found in DataFrame")
    
#     model = glda.GuidedLDA(n_topics=5, n_iter=2000, random_state=7, refresh=20,alpha=0.01,eta=0.01)
#     print(model)
#     return
    



# def _create_lda_model_with_seed(df: pd.DataFrame, col: str = "cons_processed", topics: int = 10, seed_words: list = None) -> LdaModel:
#     """
#     Create an LDA model with a given set of words assigned to one topic.
    
#     Parameters:
#         df (pd.DataFrame): DataFrame containing the text data.
#         col (str): Column name containing preprocessed text.
#         topics (int): Number of topics for LDA.
#         seed_words (list): List of words that should be associated with a specific topic.

#     Returns:
#         LdaModel: Trained LDA model.
#     """
#     if col not in df.columns:
#         raise ValueError(f"Column '{col}' not found in DataFrame")

#     # Convert each row into a list of words (assuming space-separated tokens)
#     doc_clean = [row.split() for row in df[col]]

#     # Create dictionary
#     dictionary = Dictionary(doc_clean)

#     # Ensure seed words are in the dictionary
#     if seed_words:
#         for word in seed_words:
#             dictionary.doc2bow([word], allow_update=True)

#     # Filter extremes
#     dictionary.filter_extremes(no_below=10, no_above=0.33, keep_n=1000)

#     # Create document-term matrix
#     doc_term_matrix = [dictionary.doc2bow(doc) for doc in doc_clean]

#     # Train LDA model
#     lda_model = LdaModel(doc_term_matrix, num_topics=topics, id2word=dictionary, passes=3)

#     return lda_model



import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from gensim import corpora
from gensim.models import LdaModel

def train_lda(df: pd.DataFrame, num_topics=5, num_words=10, seed_words_per_topic=None):


    # Create a dictionary from the processed documents
    # Make a list of words for each row in the DataFrame
    df["cons_processed"] = df["cons_processed"].apply(lambda x: x.split())
    processed_docs = df["cons_processed"].tolist()
    print(processed_docs)
    dictionary = corpora.Dictionary(processed_docs)

    # Create a corpus
    corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # Create a dictionary with seed words
    seed_dict = {}
    for t_id, seed_words in enumerate(seed_words_per_topic):
        for word in seed_words:
            if word in dictionary.token2id:
                seed_dict[dictionary.token2id[word]] = t_id

    # Train the LDA model
    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=len(seed_words_per_topic),
        passes=15,
        alpha='auto',
        eta='auto',
        random_state=42
    )

    # Assign topics to documents
    doc_topics = [lda_model.get_document_topics(doc) for doc in corpus]

        # Display the topics
    for topic_id, topic in lda_model.show_topics(formatted=False, num_topics=len(seed_words_per_topic)):
        print(f"Topic #{topic_id + 1}:")
        print(" ".join([word[0] for word in topic]))
        print()
    
        # Assign topics to documents
    df['topic'] = [max(doc, key=lambda x: x[1])[0] for doc in doc_topics]

    print(df[['cons_processed', 'topic']].head())


    return lda_model


# Print the resulting topics
df = _load_data._quick_load("data.pkl", filetype="pkl")
df = _preprocess_column(df, col="cons")
seed_words = _load_data._quick_load("bureaucratic_words.pkl", filetype="pkl")
seed_words = seed_words[seed_words["count"] > 1]
seed_words = [seed_words["word"].tolist()]
train_lda(df, num_topics=5, seed_words_per_topic=seed_words)


print(df)


