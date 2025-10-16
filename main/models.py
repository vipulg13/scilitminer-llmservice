import json
import logging
import os
from abc import ABC, abstractmethod, abstractproperty
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    TypedDict,
    TypeVar,
    Union,
)

import polars as pl
from openai import Client
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
    completion_create_params,
)
from openai.types.completion import Completion, CompletionUsage

logger = logging.getLogger(__name__)

OpenAIResponse = TypeVar("OpenAIResponse", Completion, ChatCompletion)

MODEL_CAPABILITIES = {
    "gpt-4": {"temperature", "max_tokens"},
    "gpt-4o": {"temperature", "max_tokens"},
    "gpt-4o-mini": {"temperature", "max_tokens"},
    "o3-mini": {"max_completion_tokens"},
    "o4-mini": {"max_completion_tokens"},
    "gpt-4.1": {"max_completion_tokens", "temperature"},
    "gpt-5-chat": {"max_completion_tokens"},
    "gpt-5": {"max_completion_tokens"},
    "Llama-3.3-70B-Instruct": {"max_tokens", "temperature"},
    "DeepSeek-V3-0324": {"max_tokens", "temperature"},
    "gpt-oss-120b": {"max_tokens", "temperature"},
}

def filter_chat_params(model_name: str, raw_params: Dict[str, Any]) -> Dict[str, Any]:
    required_keys = {"model", "messages", "response_format", "seed"}
    allowed = MODEL_CAPABILITIES.get(model_name, set())
    return {
        k: v for k, v in raw_params.items()
        if k in allowed or k in required_keys
    }


class ResponseCost(TypedDict):
    total_prompt_tokens: int
    total_completion_tokens: int
    price_of_prompt: float
    price_of_completion: float
    total_price: float


class Model(ABC):
    """Abstract base class for models.

    Parameters
    ---------
    name : str | None
        The name of the model instance

    Attributes
    ----------
    context_length : int
        The maximum number of tokens the model can handle.
    max_tokens : int
        The maximum number of tokens to generate.
    temperature : float
        Sampling temperature to use for generation.
    stop_sequences : List[str]
        List of sequences where the generation will stop.
    """
    context_length = 0
    max_tokens = 2048
    max_completion_tokens = 4096
    temperature: float = 0
    stop_sequences: List[str] = []

    def __init__(self, name: str | None = None):
        self.name = name or "Undefined"
        self.usage: List[CompletionUsage] = []

    def get_name(self) -> str:
        """
        Get the name of the model.

        Returns
        -------
        str
            The name of the model.
        """
        return self.name


class ChatModel(Model):
    """Abstract base class for chat models."""

    @abstractmethod
    def get_response(
        self,
        messages: List[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        ...


class CompletionModel(Model):
    """Abstract base class for completion models."""

    @abstractmethod
    def get_response(self, prompt: str) -> Completion:
        ...
        client: Client | None = None,


class OpenAIModel(Model):
    """Base class for models using OpenAI's API.

    Parameters
    ----------
    name : str
        The name of the model instance.
    client : Client, optional
        An OpenAI client for managing the session.
        If not provided, an AzureOpenAI client is used by default.
    """

    def __init__(
        self,
        client: Client,
        name: str,
    ):
        self.client = client
        super().__init__(name=name)


class OpenAIChatCompletion(OpenAIModel, ChatModel):
    """A class for managing chat completions using OpenAI's API.

    Extends the OpenAIModel to provide chat completion functionality.

    Parameters
    ----------
    client : Client, optional
        An OpenAI client for managing the session.
        If not provided, an AzureOpenAI client is used by default.
    """

    def __init__(
        self,
        client: Client,
        response_format: completion_create_params.ResponseFormat | None = None,
    ):
        if response_format is None:
            response_format = {"type": "text"}
        self.response_format = response_format
        super().__init__(client=client, name=self.name)

    def get_response(
        self,
        messages: List[ChatCompletionMessageParam],
    ) -> ChatCompletion:
        """Fetches a chat completion response from the OpenAI API.

        Returns
        -------
        ChatCompletion
            The chat completion response from the OpenAI API.
        """
        logger.debug(f"OpenAI query: {messages}")

        raw_params = {
            "model": self.name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": getattr(self, "max_tokens", None),
            "max_completion_tokens": getattr(self, "max_completion_tokens", None),
            "response_format": self.response_format,
            "seed": os.environ.get("seed", 66),
        }

        # Remove None values and filter by model capability
        filtered_params = filter_chat_params(self.name, raw_params)
        response = self.client.chat.completions.create(**filtered_params)

        return response

class gpt35turbo(OpenAIChatCompletion):
    context_length = 16385

    def __init__(
        self,
        name="gpt-35-turbo",
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        client: Client | None = None,
        max_tokens=2048,
        temperature: int = 0,
    ):
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__(client, response_format)


class gpt35turbo_16k(OpenAIChatCompletion):
    context_length = 16000

    def __init__(
        self,
        name="gpt-35-turbo-16k",
        client: Client | None = None,
    ):
        self.name = name
        super().__init__(client)


class gpt4(OpenAIChatCompletion):
    context_length = 65536

    def __init__(
        self,
        name="gpt-4",
        temperature=0,
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_tokens: int = 2048,
        client: Client | None = None,
    ):
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        super().__init__(client, response_format)


class gpt4_32k(OpenAIChatCompletion):
    name = "gpt-4-32k"
    context_length = 32768


class gpt4o(OpenAIChatCompletion):
    context_length = 128000

    def __init__(
        self,
        name="gpt-4o",
        temperature=0,
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_tokens: int = 4096,
        client: Client | None = None,
    ):
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        super().__init__(client, response_format)


class gpt4omini(OpenAIChatCompletion):
    context_length = 65536

    def __init__(
        self,
        name="gpt-4o-mini",
        temperature=0,
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_tokens: int = 4096,
        client: Client | None = None,
    ):
        self.name = name
        self.temperature = temperature
        self.max_tokens = max_tokens
        super().__init__(client, response_format)


class o3mini(OpenAIChatCompletion):
    context_length = 128000

    def __init__(
        self,
        name="o3-mini",
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_completion_tokens: int = 4096,
        client: Client | None = None,
    ):
        self.name = name
        self.max_completion_tokens = max_completion_tokens
        super().__init__(client, response_format)

class o4mini(OpenAIChatCompletion):
    context_length = 128000

    def __init__(
        self,
        name="o4-mini",
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_completion_tokens: int = 4096,
        client: Client | None = None,
    ):
        self.name = name
        self.max_completion_tokens = max_completion_tokens
        super().__init__(client, response_format)


class gpt41(OpenAIChatCompletion):
    context_length = 128000

    def __init__(
        self,
        name="gpt-4.1",
        temperature=0,
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_completion_tokens: int = 4096,
        client: Client | None = None,
    ):
        self.name = name
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens
        super().__init__(client, response_format)


class gpt5chat(OpenAIChatCompletion):
    context_length = 128000

    def __init__(
        self,
        name="gpt-5-chat",
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_completion_tokens: int = 8192,
        client: Client | None = None,
    ):
        self.name = name
        self.max_completion_tokens = max_completion_tokens
        super().__init__(client, response_format)

class gpt5(OpenAIChatCompletion):
    context_length = 128000

    def __init__(
        self,
        name="gpt-5",
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_completion_tokens: int = 8192,
        client: Client | None = None,
    ):
        self.name = name
        self.max_completion_tokens = max_completion_tokens
        super().__init__(client, response_format)

class Llama3370BI(OpenAIChatCompletion):
    context_length = 128000

    def __init__(
        self,
        name="Llama-3.3-70B-Instruct",
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_tokens: int = 8192,
        temperature: int = 0,
        client: Client | None = None,
    ):
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__(client, response_format)

class DeepSeekV30324(OpenAIChatCompletion):
    context_length = 128000

    def __init__(
        self,
        name="DeepSeek-V3-0324",
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_tokens: int = 8192,
        temperature: int = 0,
        client: Client | None = None,
    ):
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__(client, response_format)

class gptoss120b(OpenAIChatCompletion):
    context_length = 128000

    def __init__(
        self,
        name="gpt-oss-120b",
        response_format: completion_create_params.ResponseFormat = {"type": "json_object"},
        max_tokens: int = 8192,
        temperature: int = 0,
        client: Client | None = None,
    ):
        self.name = name
        self.max_tokens = max_tokens
        self.temperature = temperature
        super().__init__(client, response_format)




class ModelWithParams(TypedDict):
    name: str
    paramameters: Dict[str, Any]
    


DEModelName = Literal["gpt-35-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "o3-mini", "o4-mini", "gpt-4.1", "gpt-5-chat", "gpt-5", "Llama-3.3-70B-Instruct", "DeepSeek-V3-0324", "gpt-oss-120b"]

DEModel = Union[DEModelName, ModelWithParams]


def get_chat_model(
    name: DEModel = "gpt-4",
    temperature: float = 0,
    mode: Literal["text", "json_object", "json_schema"] = "json_schema",   # changed for test gpt-5 gpt4 family supports json_schema
    max_tokens: int = 4096,
    max_completion_tokens: int = 4096,
    client: Client | None = None,
    json_schema: dict | None = None
) -> ChatModel:
    
    if mode == "json_schema":
        # If mode is 'json_schema', add schema info to the response_format
        if json_schema is None:
            raise ValueError("json_schema must be provided when mode is 'json_schema'")
        response_format = {
            "type": "json_schema",
            "json_schema": json_schema, 
        }
    else:
        response_format = {"type": mode} 

    params = dict(
        response_format=response_format,
        max_tokens=max_tokens,
        temperature=temperature,
        max_completion_tokens=max_completion_tokens,
    )

    model_mapping = {"gpt-35-turbo": gpt35turbo, "gpt-4": gpt4, "gpt-4o": gpt4o, "gpt-4o-mini": gpt4omini, "o3-mini": o3mini, "o4-mini": o4mini, "gpt-4.1":gpt41, "gpt-5-chat":gpt5chat, "gpt-5":gpt5, "Llama-3.3-70B-Instruct":Llama3370BI, "DeepSeek-V3-0324":DeepSeekV30324, "gpt-oss-120b":gptoss120b}

    if isinstance(name, Dict):
        model_name = name["name"]
        if parameters := name.get("parameters"):
            for key, value in parameters.items():
                if key == "mode":
                    if value == "json_schema":
                        params["response_format"] = {
                            "type": "json_schema",
                            "json_schema": parameters.get("json_schema", {}),
                        }
                    else:
                        params["response_format"]["type"] = value
                else:
                    params[key] = value
    else:
        model_name = name


    obj = model_mapping[model_name]
    deployment_name = model_name

    client_obj = Client if client is None else client

    constructor_params = filter_chat_params(deployment_name, params)
    return obj(deployment_name, client=client_obj, **constructor_params)
    #return obj(deployment_name, client=client_obj, **params)
