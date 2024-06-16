from typing import Any, Generator, List, Optional

import openai
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel

from .base import CompletionService, EmbeddingService
from .util import ChatMessageType, format_chat_message

DEFAULT_STOP_TOKEN: List[str] = ["<EOS>"]


class OpenAIConfig(BaseModel):
    api_type: str = "openai"
    api_base: str = "https://api.openai.com/v1"
    api_key: str
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-ada-002"
    response_format: str = "json_object"
    api_version: str = "2023-12-01-preview"
    api_auth_type: str = "openai"
    stop_token: List[str] = DEFAULT_STOP_TOKEN
    temperature: float = 0
    max_tokens: int = 1024
    top_p: float = 0
    frequency_penalty: float = 0
    presence_penalty: float = 0
    seed: int = 123456


class OpenAIService(CompletionService, EmbeddingService):
    config: OpenAIConfig

    def __init__(self, api_key: str):
        self.config = OpenAIConfig(api_key=api_key)
        self.client: OpenAI = OpenAI(
            base_url=self.config.api_base,
            api_key=self.config.api_key,
        )

    def chat_completion(
        self,
        messages: List[ChatMessageType],
        stream: bool = True,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Generator[ChatMessageType, None, None]:
        engine = self.config.model

        temperature = (
            temperature if temperature is not None else self.config.temperature
        )
        max_tokens = max_tokens if max_tokens is not None else self.config.max_tokens
        top_p = top_p if top_p is not None else self.config.top_p
        stop = stop if stop is not None else self.config.stop_token
        seed = self.config.seed

        try:
            tools_kwargs = {}
            if "tools" in kwargs and "tool_choice" in kwargs:
                tools_kwargs["tools"] = kwargs["tools"]
                tools_kwargs["tool_choice"] = kwargs["tool_choice"]
            if "response_format" in kwargs:
                response_format = kwargs["response_format"]
            elif self.config.response_format == "json_object":
                response_format = {"type": "json_object"}
            else:
                response_format = None

            res: Any = self.client.chat.completions.create(
                model=engine,
                messages=messages,  # type: ignore
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                stop=stop,
                stream=stream,
                seed=seed,
                response_format=response_format,
                **tools_kwargs,
            )
            if stream:
                role: Any = None
                for stream_res in res:
                    if not stream_res.choices:
                        continue
                    delta = stream_res.choices[0].delta
                    if delta is None:
                        continue

                    role = delta.role if delta.role is not None else role
                    content = delta.content if delta.content is not None else ""
                    if content is None:
                        continue
                    yield format_chat_message(role, content)
            else:
                oai_response = res.choices[0].message
                if oai_response is None:
                    raise Exception("OpenAI API returned an empty response")
                response: ChatMessageType = format_chat_message(
                    role=(
                        oai_response.role
                        if oai_response.role is not None
                        else "assistant"
                    ),
                    message=(
                        oai_response.content if oai_response.content is not None else ""
                    ),
                )
                if oai_response.tool_calls is not None:
                    import json

                    response["role"] = "function"
                    response["content"] = json.dumps(
                        [
                            {
                                "name": t.function.name,
                                "arguments": json.loads(t.function.arguments),
                            }
                            for t in oai_response.tool_calls
                        ],
                    )
                yield response

        except openai.APITimeoutError as e:
            # Handle timeout error, e.g. retry or log
            raise Exception(f"OpenAI API request timed out: {e}")
        except openai.APIConnectionError as e:
            # Handle connection error, e.g. check network or log
            raise Exception(f"OpenAI API request failed to connect: {e}")
        except openai.BadRequestError as e:
            # Handle invalid request error, e.g. validate parameters or log
            raise Exception(f"OpenAI API request was invalid: {e}")
        except openai.AuthenticationError as e:
            # Handle authentication error, e.g. check credentials or log
            raise Exception(f"OpenAI API request was not authorized: {e}")
        except openai.PermissionDeniedError as e:
            # Handle permission error, e.g. check scope or log
            raise Exception(f"OpenAI API request was not permitted: {e}")
        except openai.RateLimitError as e:
            # Handle rate limit error, e.g. wait or log
            raise Exception(f"OpenAI API request exceeded rate limit: {e}")
        except openai.APIError as e:
            # Handle API error, e.g. retry or log
            raise Exception(f"OpenAI API returned an API Error: {e}")

    def get_embeddings(self, strings: List[str]) -> List[List[float]]:
        embedding_results = self.client.embeddings.create(
            input=strings,
            model=self.config.embedding_model,
        ).data
        return [r.embedding for r in embedding_results]
