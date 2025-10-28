import json
from typing import Any, AsyncGenerator, Generator, Iterable, Optional, Union

import litellm
from litellm import Message
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ToolParam,
)
from google.adk.models.lite_llm import (
    LiteLLMClient,
    LiteLlm,
    _build_request_log,
    _get_completion_inputs,
)
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from litellm.types.llms.openai import ResponsesAPIResponse
from pydantic import Field
from typing_extensions import override
from veadk.utils.logger import get_logger


# This will add functions to prompts if functions are provided.
litellm.add_function_to_prompt = True

logger = get_logger(__name__)


def _responses_api_response_to_model_response(
    response: ResponsesAPIResponse,
) -> LlmResponse:
    """
    Convert ResponsesAPIResponse to LlmResponse.

    Args:
        response: The response from the responses API

    Returns:
        LlmResponse object compatible with the existing interface
    """
    parts = []

    # Process output items
    if hasattr(response, "output") and response.output:
        for output_item in response.output:
            # Handle both object and dict formats
            item_type = (
                getattr(output_item, "type", None)
                if hasattr(output_item, "type")
                else output_item.get("type")
                if isinstance(output_item, dict)
                else None
            )

            # Handle message content
            if item_type == "message":
                # Get content from either object attribute or dict key
                content = (
                    getattr(output_item, "content", None)
                    if hasattr(output_item, "content")
                    else output_item.get("content")
                    if isinstance(output_item, dict)
                    else None
                )

                if content:
                    for content_item in content:
                        # Handle both object and dict formats for content items
                        content_type = (
                            getattr(content_item, "type", None)
                            if hasattr(content_item, "type")
                            else content_item.get("type")
                            if isinstance(content_item, dict)
                            else None
                        )
                        content_text = (
                            getattr(content_item, "text", None)
                            if hasattr(content_item, "text")
                            else content_item.get("text")
                            if isinstance(content_item, dict)
                            else None
                        )

                        # Handle text content
                        if content_type == "output_text" and content_text:
                            parts.append(types.Part.from_text(text=content_text))

            # Handle reasoning items (skip for now)
            elif item_type == "reasoning":
                pass

            # Handle function tool calls
            elif item_type == "function_tool_call":
                # Get function from either object attribute or dict key
                function = (
                    getattr(output_item, "function", None)
                    if hasattr(output_item, "function")
                    else output_item.get("function")
                    if isinstance(output_item, dict)
                    else None
                )

                if function:
                    function_name = (
                        getattr(function, "name", None)
                        if hasattr(function, "name")
                        else function.get("name")
                        if isinstance(function, dict)
                        else None
                    )
                    function_args = (
                        getattr(function, "arguments", None)
                        if hasattr(function, "arguments")
                        else function.get("arguments")
                        if isinstance(function, dict)
                        else None
                    )

                    if function_name:
                        part = types.Part.from_function_call(
                            name=function_name,
                            args=json.loads(function_args or "{}"),
                        )
                        # Get ID from either object attribute or dict key
                        item_id = (
                            getattr(output_item, "id", None)
                            if hasattr(output_item, "id")
                            else output_item.get("id")
                            if isinstance(output_item, dict)
                            else None
                        )
                        if item_id:
                            part.function_call.id = item_id
                        parts.append(part)

    # Create LlmResponse
    llm_response = LlmResponse(
        content=types.Content(role="model", parts=parts), partial=False
    )

    # Add usage metadata if available
    if hasattr(response, "usage") and response.usage:
        # Map the usage fields correctly
        input_tokens = getattr(response.usage, "input_tokens", 0)
        output_tokens = getattr(response.usage, "output_tokens", 0)
        total_tokens = getattr(
            response.usage, "total_tokens", input_tokens + output_tokens
        )

        llm_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=input_tokens,
            candidates_token_count=output_tokens,
            total_token_count=total_tokens,
        )

    return llm_response


class ArkLLMClient(LiteLLMClient):
    async def aresponses(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        tools: Optional[Iterable[ToolParam]] = None,
        **kwargs,
    ) -> Union[
        ResponsesAPIResponse, BaseResponsesAPIStreamingIterator
    ]:  # todo: 流式类型还没有测试
        return await litellm.aresponses(
            model=model,
            input=input,
            tools=tools,
            **kwargs,
        )

    def responses(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        tools: Optional[Iterable[ToolParam]] = None,
        **kwargs,
    ) -> Union[ResponsesAPIResponse, BaseResponsesAPIStreamingIterator]:
        return litellm.responses(
            model=model,
            input=input,
            tools=tools,
            **kwargs,
        )


def _messages_to_responses_input(messages: list[Message]) -> list[dict]:
    """Convert litellm messages to responses API input format."""
    input_items = []

    for message in messages:
        # Convert each message to EasyInputMessageParam format
        input_item = {
            "type": "message",
            "role": message.get("role", "user"),
            "content": message.get("content", ""),
        }
        input_items.append(input_item)

    return input_items


class ArkLlm(LiteLlm):
    llm_client: ArkLLMClient = Field(default_factory=ArkLLMClient)
    _additional_args: dict[str, Any] = None

    def __init__(self, model: str, **kwargs):
        super().__init__(model=model, **kwargs)

    @override
    def generate_content(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> Generator[LlmResponse, None, None]:
        """Generates content synchronously.

        Args:
          llm_request: LlmRequest, the request to send to the LiteLlm model.
          stream: bool = False, whether to do streaming call.

        Yields:
          LlmResponse: The model response.
        """

        self._maybe_append_user_content(llm_request)
        logger.debug(_build_request_log(llm_request))

        messages, tools, response_format, generation_params = _get_completion_inputs(
            llm_request
        )

        if "functions" in self._additional_args:
            # LiteLLM does not support both tools and functions together.
            tools = None

        # Convert messages to responses API input format
        input_param = _messages_to_responses_input(messages)

        completion_args = {
            "model": self.model,
            "input": input_param,
            "tools": tools,
            "response_format": response_format,
        }
        completion_args.update(self._additional_args)

        if generation_params:
            completion_args.update(generation_params)

        if stream:
            raise NotImplementedError("LiteLlm does not support streaming call.")
        else:
            response = self.llm_client.responses(**completion_args)
            yield _responses_api_response_to_model_response(response)

    @override
    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Generates content asynchronously.

        Args:
          llm_request: LlmRequest, the request to send to the LiteLlm model.
          stream: bool = False, whether to do streaming call.

        Yields:
          LlmResponse: The model response.
        """

        self._maybe_append_user_content(llm_request)
        logger.debug(_build_request_log(llm_request))

        messages, tools, response_format, generation_params = _get_completion_inputs(
            llm_request
        )

        if "functions" in self._additional_args:
            # LiteLLM does not support both tools and functions together.
            tools = None

        # Convert messages to responses API input format
        input_param = _messages_to_responses_input(messages)

        completion_args = {
            "model": self.model,
            "input": input_param,
            "tools": tools,
            "response_format": response_format,
        }
        completion_args.update(self._additional_args)

        if generation_params:
            completion_args.update(generation_params)

        if stream:
            raise NotImplementedError("LiteLlm does not support streaming call.")
        else:
            response = await self.llm_client.aresponses(**completion_args)
            yield _responses_api_response_to_model_response(response)

    @classmethod
    @override
    def supported_models(cls) -> list[str]:
        """Provides the list of supported models.

        LiteLlm supports all models supported by litellm. We do not keep track of
        these models here. So we return an empty list.

        Returns:
          A list of supported models.
        """

        return []
