import os

from azure_client import setup_azure_open_ai
from openai import AzureOpenAI, AsyncOpenAI

os.environ["OPENAI_API_KEY"] = 'xxx'  # replace with your actual key
os.environ["OPENAI_ENDPOINT"] = 'xxx'  # replace with your actual endpoint
os.environ["OPENAI_API_VERSION"] = '2025-04-01-preview'

os.environ["AZURE_OPENSOURCE_API_KEY"] = 'xxx'  # replace with your actual endpoint
os.environ["AZURE_OPENSOURCE_ENDPOINT"] = 'xxx'  # replace with your actual endpoint
os.environ["AZURE_OPENSOURCE_API_VERSION"] = '2024-05-01-preview'



def create_client(type: str = "openai"):
    if type == "openai":
        setup_azure_open_ai(api_base=os.environ["OPENAI_ENDPOINT"], 
                            api_key=os.environ["OPENAI_API_KEY"], 
                            api_version=os.environ["OPENAI_API_VERSION"])
        return AzureOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                           azure_endpoint=os.environ["OPENAI_ENDPOINT"],
                           api_version=os.environ["OPENAI_API_VERSION"],
                           timeout=180)
    elif type == "opensource":
        setup_azure_open_ai(api_base=os.environ["AZURE_OPENSOURCE_ENDPOINT"], 
                            api_key=os.environ["AZURE_OPENSOURCE_API_KEY"], 
                            api_version=os.environ["AZURE_OPENSOURCE_API_VERSION"])
        return AzureOpenAI(api_key=os.environ["AZURE_OPENSOURCE_API_KEY"],
                            azure_endpoint=os.environ["AZURE_OPENSOURCE_ENDPOINT"],
                            api_version=os.environ["AZURE_OPENSOURCE_API_VERSION"],
                            timeout=180)

