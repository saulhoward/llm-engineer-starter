import abc
from typing import Any, Generator, List, Optional

from .util import ChatMessageType


class CompletionService(abc.ABC):
    @abc.abstractmethod
    def chat_completion(
        self,
        messages: List[ChatMessageType],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Generator[ChatMessageType, None, None]:
        """
        Chat completion API

        :param messages: list of messages
        :param temperature: temperature
        :param max_tokens: maximum number of tokens
        :param top_p: top p

        :param kwargs: other model specific keyword arguments

        :return: generator of messages
        """

        raise NotImplementedError
