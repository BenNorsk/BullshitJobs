from bullshit_jobs.load_data._load_data import _quick_load
import re
import json
from typing import Optional
import requests
import pathlib


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
        bs_score = json_answer["bs_score"]
        print(f'The answer is: {answer}')
        print(f'bs_score: {bs_score}')

    except:
        print("Parsing the answer to the JSON to extract the score failed. The answer which failed was:")
        print(answer)

    return bs_score


def _send_text_to_llm(text: str) -> int:
    """
    Sends a text to the LLM model and returns the bullshit score.

    Parameters:
    -----------
    text: str
        The text to send to the LLM model.

    Returns:
    --------
    bs_score: int
        The bullshit score.
    """

    # Make this into a python request
    url = "http://localhost:11434/api/chat"
    data = {
        "model": "llama3.2",
        "messages": [
            { "role": "user", "content": text }
        ],
        "stream": False
    }
    # Set the headers
    headers = {"Content-Type": "application/json"}
    # Send the request
    response = requests.post(url, headers=headers, json=data)
    print(response)
    answer = json.loads(response.text)["message"]["content"]
    print(f'The answer is: {answer}')
    bs_score = _extract_text_from_answer(answer)
    return bs_score


def _create_llm_bs_score():
    """
    Create a bullshit score using the LLM model.
    """

    # LLM Prompt Text
    # Read in the llm_prompt.txt file
    path = pathlib.Path(__file__).parent.parent.parent / "bullshit_jobs" / "bullshit_score" / "llm_prompt.txt"
    with open(path, "r") as file:
        llm_prompt = file.read()

    # Load in the review
    review = "Lots of red tape to cut through"

    # Replace '{{REVIEW}}' with the review
    llm_prompt = llm_prompt.replace("{{REVIEW}}", review)

    print("----- Submitted LLM Prompt -----")
    print(llm_prompt)

    # Send the text to the LLM model
    bs_score = _send_text_to_llm(llm_prompt)

    return

if __name__ == "__main__":
    _create_llm_bs_score()
