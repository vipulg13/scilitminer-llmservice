import requests
import json
import os
from types import SimpleNamespace


def blablador_get_models():
    url = f"{os.environ['BLABLADOR_API_URL']}/v1/models"

    headers = {
        "Authorization": f"Bearer {os.environ['BLABLADOR_API_KEY']}",
        "Content-Type": "application/json",
    }

    
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    modelnames_object = response.json()
    models = [model["id"] for model in modelnames_object.get("data", [])]
    return models


# Function to send a chat completion request to the Blablador API
def blablador_chat_completion(model_name, system_message, max_tokens=2048, temperature=0.2):
    """
    Sends a chat completion request to the specified API.

    Parameters:
        model_name (str): Name of the model to use.
        system_message (str): Content of the system message.
        max_tokens (int): Maximum number of tokens to generate. Default is 2048.

    Returns:
        Raw response from the API.
    """
    url = f"{os.environ['BLABLADOR_API_URL']}/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['BLABLADOR_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": system_message},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }

    response = requests.post(url, headers=headers, json=payload)
    if response.status_code == 200:
        response_str = response.content.decode('utf-8')
        response = json.loads(response_str, object_hook=lambda d: SimpleNamespace(**d))
    else:
        raise ValueError(f"Error in API request: {response.status_code} - {response.text}")
    return response