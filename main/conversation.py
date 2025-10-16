import json
from typing import List, Optional

import polars as pl
from agents import Agent
from models import get_chat_model
from openai import AzureOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


def azure_safe_message(response_obj: ChatCompletionMessage):
    dict_response = response_obj.model_dump()
    dict_response.pop("tool_calls")  # to make azure happy
    if dict_response.get("function_call") is None:
        dict_response.pop("function_call")
    return dict_response


class Conversation(object):
    """A class to manage a conversation with an AI agent.

    Parameters
    ----------
    agent : Agent | None, optional
        The AI agent to converse with. If not provided, a default Agent will be created.

    Attributes
    ----------
    agent : Agent
        The AI agent used for the conversation.
    messages : List[ChatCompletionMessageParam]
        The list of messages in the conversation.
    """

    messages: List[ChatCompletionMessageParam]

    def __init__(
        self,
        agent: Agent | None = None,
        context: str | None = None,
    ):
        if agent is None:
            self.agent = Agent()
        elif isinstance(agent, Agent):
            self.agent = agent
        else:
            raise ValueError(f"Agent {agent} type incompatible ({type(agent)})")
        self.context = context

        if self.agent.system_message is not None:
            self.messages = [
                ChatCompletionSystemMessageParam(
                    role="system", content=self.agent.system_message
                )
            ]
        else:
            self.messages: List[ChatCompletionMessageParam] = []

    def run_once(self, question: Optional[str] = None) -> ChatCompletion:
        """Performs one interaction with the agent, optionally starting with a user question.

        Parameters
        ----------
        question : Optional[str]
            The user's question to start the conversation with.
            If None, the conversation continues with the agent's response.

        Returns
        -------
        ChatCompletion
            The agent's response as a ChatCompletion object.
        """
        if question is not None:
            self.add_messages(
                [ChatCompletionUserMessageParam(role="user", content=question)]
            )
        response = self.agent.call(self.messages)
        response_obj = response.choices[0].message
        self.messages.append(azure_safe_message(response_obj))
        return response