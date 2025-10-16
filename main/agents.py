from abc import ABC
from typing import List

from models import ChatModel, get_chat_model
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessageParam,
)


class Agent(ABC):
    def __init__(
        self,
        model: ChatModel | None = None,
        system_message: str | None = None,
        name="Unnamed Agent",
    ):
        self.model: ChatModel = model or get_chat_model()
        self.system_message = system_message
        self.name = name

    def __repr__(self):
        return f"Agent with using {self.model.name} with system message:'{self.system_message}'"

    def call(self, messages: List[ChatCompletionMessageParam]) -> ChatCompletion:
        response = self.model.get_response(messages=messages)
        return response
