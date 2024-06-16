import types
import os
from typing import Any, List, Optional

from .base import (
    CompletionService,
)
from .ollama import OllamaService
from .openai import OpenAIService
from .util import ChatMessageType, format_chat_message

llm_completion_config_map = {
    "openai": OpenAIService,
    "ollama": OllamaService,
}


class LLMApi(object):
    completion_service: CompletionService
    api_type: str

    def __init__(
        self,
        api_type: str = "openai",
    ):
        self.api_type = api_type

        if api_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            self.completion_service = OpenAIService(api_key=api_key)
        elif api_type == "ollama":
            self.completion_service = OllamaService()
        else:
            raise ValueError(f"API type {api_type} is not supported")

    def chat_completion(
        self,
        messages: List[ChatMessageType],
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> ChatMessageType:
        msg: ChatMessageType = format_chat_message("assistant", "")
        completion_service = self.completion_service
        for msg_chunk in completion_service.chat_completion(
            messages,
            stream,
            temperature,
            max_tokens,
            top_p,
            stop,
            **kwargs,
        ):
            msg["role"] = msg_chunk["role"]
            msg["content"] += msg_chunk["content"]
            if "name" in msg_chunk:
                msg["name"] = msg_chunk["name"]
        return msg
