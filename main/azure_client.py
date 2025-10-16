from typing import Literal
import os

def setup_azure_open_ai(
    api_base: str = None,
    api_version: Literal["2024-12-01-preview"] = "2024-12-01-preview",
    api_key: str = None,
):
    try:
        import openai
    except ImportError:
        raise ImportError(
            "Can't find openai. Install through `pip install --upgrade openai."
        )

    if int(openai.version.VERSION.split(".")[0]) < 1:
        openai.api_key = api_key
        openai.api_base = api_base
        openai.api_version = api_version
        openai.api_type = "azure"
        return None