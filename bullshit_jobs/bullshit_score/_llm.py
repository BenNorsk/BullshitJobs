from bullshit_jobs.load_data._load_data import _quick_load, _save_data
import re
import json
from typing import Optional
import requests
import pathlib
import pandas as pd
from openai import OpenAI, ChatCompletion
import openai
import copy
import os
from dotenv import load_dotenv
import sys


def _extract_text_from_answer(answer: str) -> Optional[str]:
    """
    Extracts the text from an answer.

    Parameters:
    -----------
    answer: str
        The answer to extract the text from.

    Returns:
    --------
    text: str
        The extracted text.
    """
    # Replace all \ and \n in the response
    answer = answer.replace("\\", "")
    answer = answer.replace("\\n", "")

    # Find the text between the curly braces
    pattern = r'\{[^{}]*\}'
    answer = re.search(pattern, answer).group()

    # Try to load the text as a JSON
    bs_score = None
    try:
        json_answer = json.loads(answer)
        if "bs_score" in json_answer:
            bs_score = json_answer["bs_score"]
            bs_score = float(bs_score)
        elif "bs_ score" in json_answer:
            bs_score = json_answer["bs_ score"]
            bs_score = float(bs_score)
        elif "bs__score" in json_answer:
            bs_score = json_answer["bs__score"]
            bs_score = float(bs_score)
        else:
            # Convert json answer to string
            text = str(json_answer)
            pattern = r'[-+]?\d*\.\d+|\d+'
            match = re.search(pattern, text)
            if match:
                number = float(match.group())  # Convert to float if needed
                bs_score = number
        print(f'The answer is: {answer}')
        print(f'bs_score: {bs_score}')

    except:
        print("Parsing the answer to the JSON to extract the score failed. The answer which failed was:")
        print(answer)

    return bs_score


# def _send_text_to_llm_deepseek(text: str):
#     """
#     Sends a text to the LLM model and returns the bullshit score.

#     Parameters:
#     -----------
#     text: str
#         The text to send to the LLM model.

#     Returns:
#     --------
#     response: requests.models.Response
#         The response from the LLM model
#     """
#     # API Key
#     api_key = ""

#     # Make this into a python request
#     client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

#     # Set the message
#     messages = [{"role": "user", "content": text}]
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=messages
#     )

#     # Retrieve the response
#     response = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=messages
#     )

    
#     content = response.choices[0].message.content

#     return content



def _send_text_to_llm_openai(text: str, nr: str):
    """
    Sends a text to the OpenAI GPT-3.5 Turbo model and returns the bullshit score.

    Parameters:
    -----------
    text: str
        The text to send to the LLM model.

    Returns:
    --------
    response: str
        The response from the LLM model
    """
    if nr is None:
        nr = "1"

    # Retrieve the API key from the environment variables
    api_key = os.getenv(f'OPENAI_API_KEY_{nr}')

    if not api_key:
        raise ValueError(f'OPENAI_API_KEY_{nr} not found in environment variables.')
    
    # Set the OpenAI API key
    openai.api_key = api_key
    
    # Prepare the message payload
    messages = [{"role": "user", "content": text}]
    
    client = OpenAI(
    api_key=api_key,  # This is the default and can be omitted
    )   

    chat_completion = client.chat.completions.create(
        messages=messages,
        model="gpt-4o-mini",
    )

    # Extract and return the response content
    content = chat_completion.choices[0].message.content
    print(content)
    
    return content

def _get_new_bs_score_llm(review: str, nr: str) -> int:
    
    # # Clear the LLM model
    # response = _send_text_to_llm("/clear")
    # print(response.text)

    # LLM Prompt Text
    response = _send_text_to_llm_openai(review, nr)
    
    try:
        print("----- Answer -----")
        print(response)
        bs_score = _extract_text_from_answer(response)
        print("----- BS Score -----")
        print(bs_score)
    except:
        print("The response from the LLM model was not in the expected format.")
        print(response)
        bs_score = None
    
    return bs_score


def _create_llm_bs_score(nr: Optional[str] = None):
    """
    Create a bullshit score using the LLM model.
    """

    # LLM Prompt Text
    # Read in the llm_prompt.txt file
    path = pathlib.Path(__file__).parent.parent.parent / "bullshit_jobs" / "bullshit_score" / "llm_prompt.txt"
    with open(path, "r") as file:
        llm_prompt = file.read()  

    # Load the data
    if nr is None:
        df = _quick_load("data_with_bs_score.pkl")
    else:
        df = _quick_load(f"llm/data_with_bs_score_{nr}.pkl")

    # Load the save counter 
    i = 0

    # Iterate over all observations
    for index, row in df.iterrows():
        review = row["cons"]
        review_id = row["review_id"]
        bs_score = row["bs_score_llm"]

        try:
            bs_score = float(bs_score)
        except:
            bs_score = None

        print(bs_score)

        if not (isinstance(bs_score, float)):
            print("----- Review -----")
            print(review)

            # Copy the llm_prompt
            llm_prompt_copy = copy.copy(llm_prompt)

            # Replace '{{REVIEW}}' with the review
            llm_prompt_copy = llm_prompt_copy.replace("{{REVIEW}}", review)
            print("----- LLM Prompt -----")
            print(llm_prompt_copy)
            
            # Send the text to the LLM model
            try:
                bs_score = _get_new_bs_score_llm(llm_prompt_copy, nr)
            except:
                print(f'The bs_score_llm is None. Skipping this review.')
                continue

            # Increase the save counter by one
            i += 1
            print(f'Counter: {i}')

            if bs_score is None:
                print(f'The bs_score_llm is None. Skipping this review.')
                continue

            else:
                # Save the bs_score to the dataframe at the review id
                df.loc[df["review_id"] == review_id, "bs_score_llm"] = bs_score

                if i >= 60:
                    print("Saving the data.")
                    if nr is None:
                        # Save the data
                        _save_data(df, "data_with_bs_score")
                    else:
                        _save_data(df, f"llm/data_with_bs_score_{nr}")
                    
                    # Reset the save counter
                    i = 0
        else:
            print(f'The bs_score_llm is already observed ({bs_score}). Skipping this review.')
    if nr is None:
        # Save the data
        _save_data(df, "data_with_bs_score")
    else:
        _save_data(df, f"llm/data_with_bs_score_{nr}")
    return

def _split_data_into_chunks():
    """
    Split the data into three chunks.
    """
    raise ValueError("This function is dangerous to run. It will overwrite the data files.")

    # Load and remove all data
    df = _quick_load("data_with_bs_score.pkl")

    # Get the length of the data
    length = len(df)
    chunk_size = length // 3

    # Split the data into three chunks
    df1 = df.iloc[:chunk_size]
    df2 = df.iloc[chunk_size:2*chunk_size]
    df3 = df.iloc[2*chunk_size:]

    # Save the data DANGEROURS!
    # _save_data(df1, "llm/data_with_bs_score_1")
    # _save_data(df2, "llm/data_with_bs_score_2")
    # _save_data(df3, "llm/data_with_bs_score_3")

    return

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Find the argument to pass to the function, e.g py _llm.py 1
    try:
        number = sys.argv[1]
    
    except:
        number = None

    print(f'Number: {number}')

    _create_llm_bs_score(number)
