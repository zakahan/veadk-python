import json
from typing import Any, AsyncGenerator, Generator, Iterable, Optional, Union

import litellm
from litellm import Message
from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
from litellm.types.llms.openai import (
    ResponseInputParam,
    ToolParam,
    ChatCompletionDeveloperMessage,
)
from google.adk.models.lite_llm import (
    LiteLLMClient,
    LiteLlm,
    _build_request_log,
    _content_to_message_param,
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


class ArkLLMClient(LiteLLMClient):
    async def aresponses(
        self,
        model: str,
        input: Union[str, ResponseInputParam],
        tools: Optional[Iterable[ToolParam]] = None,
        **kwargs,
    ) -> Union[
        ResponsesAPIResponse, BaseResponsesAPIStreamingIterator
    ]:  # todo: The streaming type has not been tested yet
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


def _schema_to_responses_dict(schema: Any) -> dict:
    """Convert google.genai schema objects to JSON Schema dict for Responses API.

    Handles common fields: type, description, properties, required, enum, items.
    Recursively converts nested schemas. Falls back to model_dump/existing dicts.
    """
    if schema is None:
        return {}
    if isinstance(schema, dict):
        return schema

    # If pydantic/typed object supports model_dump, use it as a base
    base: dict = {}
    if hasattr(schema, "model_dump"):
        try:
            base = schema.model_dump(exclude_none=True)
        except Exception:
            base = {}

    # Explicitly collect common fields
    t = getattr(schema, "type", base.get("type", None))
    d = getattr(schema, "description", base.get("description", None))
    enum = getattr(schema, "enum", base.get("enum", None))
    items = getattr(schema, "items", base.get("items", None))
    properties = getattr(schema, "properties", base.get("properties", None))
    required = getattr(schema, "required", base.get("required", None))

    out: dict = {}
    if t is not None:
        out["type"] = t
    if d:
        out["description"] = d
    if enum:
        out["enum"] = list(enum)
    if items:
        out["items"] = _schema_to_responses_dict(items)
    if properties:
        out["properties"] = {
            k: _schema_to_responses_dict(v) for k, v in properties.items()
        }
    if required:
        out["required"] = list(required)

    # Merge any remaining keys from base (excluding nested already handled)
    for k, v in base.items():
        if k in out:
            continue
        if k in ("items", "properties"):
            continue
        out[k] = v

    return out


def _function_declaration_to_responses_tool(
    function_declaration: types.FunctionDeclaration,
) -> dict:
    """Converts a FunctionDeclaration to Responses API tool spec.

    Output format:
    {
      "type": "function",
      "name": str,
      "description": str,
      "parameters": { ... JSON Schema ... }
    }
    """
    assert function_declaration.name

    parameters_dict: dict = {"type": "object"}
    params = getattr(function_declaration, "parameters", None)
    if params:
        parameters_dict = _schema_to_responses_dict(params) or {"type": "object"}
        # Ensure type exists for parameters object
        if "type" not in parameters_dict:
            parameters_dict["type"] = "object"

    return {
        "type": "function",
        "name": function_declaration.name,
        "description": getattr(function_declaration, "description", "") or "",
        "parameters": parameters_dict,
    }


def _get_responses_inputs(
    llm_request: LlmRequest,
) -> tuple[
    list[Message],
    Optional[list[dict]],
    Optional[types.SchemaUnion],
    Optional[dict],
]:
    """Converts an LlmRequest to litellm inputs and extracts generation params.

    Args:
      llm_request: The LlmRequest to convert.

    Returns:
      The litellm inputs (message list, tool dictionary, response format and generation params).
    """
    # 1. Construct messages
    messages: list[Message] = []
    for content in llm_request.contents or []:
        message_param_or_list = _content_to_message_param(content)
        if isinstance(message_param_or_list, list):
            messages.extend(message_param_or_list)
        elif message_param_or_list:  # Ensure it's not None before appending
            messages.append(message_param_or_list)

    if llm_request.config.system_instruction:
        messages.insert(
            0,
            ChatCompletionDeveloperMessage(
                role="developer",
                content=llm_request.config.system_instruction,
            ),
        )

    # 2. Convert tool declarations
    tools: Optional[list[dict]] = None
    if (
        llm_request.config
        and llm_request.config.tools
        and llm_request.config.tools[0].function_declarations
    ):
        tools = [
            _function_declaration_to_responses_tool(tool)
            for tool in llm_request.config.tools[0].function_declarations
        ]

    # 3. Handle response format
    response_format: Optional[types.SchemaUnion] = None
    if llm_request.config and llm_request.config.response_schema:
        response_format = llm_request.config.response_schema

    # 4. Extract generation parameters
    generation_params: Optional[dict] = None
    if llm_request.config:
        config_dict = llm_request.config.model_dump(exclude_none=True)
        # Generate LiteLlm parameters here,
        # Following https://docs.litellm.ai/docs/completion/input.
        generation_params = {}
        param_mapping = {
            "max_output_tokens": "max_completion_tokens",
            "stop_sequences": "stop",
        }
        for key in (
            "temperature",
            "max_output_tokens",
            "top_p",
            "top_k",
            "stop_sequences",
            "presence_penalty",
            "frequency_penalty",
        ):
            if key in config_dict:
                mapped_key = param_mapping.get(key, key)
                generation_params[mapped_key] = config_dict[key]

        if not generation_params:
            generation_params = None

        return messages, tools, response_format, generation_params


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
            item_type = getattr(output_item, "type", None)

            # Handle ResponseFunctionToolCall
            if item_type == "function_call":
                function_name = getattr(output_item, "name", None)
                function_args = getattr(output_item, "arguments", None)
                call_id = getattr(output_item, "call_id", None) or getattr(
                    output_item, "id", None
                )

                if function_name:
                    # Parse arguments - handle both string and dict formats
                    if isinstance(function_args, str):
                        try:
                            args_dict = json.loads(function_args)
                        except (json.JSONDecodeError, TypeError):
                            args_dict = {}
                    elif isinstance(function_args, dict):
                        args_dict = function_args
                    else:
                        args_dict = {}

                    part = types.Part.from_function_call(
                        name=function_name,
                        args=args_dict,
                    )

                    # Set the call ID if available
                    if call_id:
                        part.function_call.id = call_id

                    parts.append(part)

            # Handle message content
            elif item_type == "message":
                content = getattr(output_item, "content", None)
                if content:
                    for content_item in content:
                        content_type = getattr(content_item, "type", None)
                        content_text = getattr(content_item, "text", None)

                        # Handle text content
                        if content_type == "output_text" and content_text:
                            parts.append(types.Part.from_text(text=content_text))

            # Handle reasoning items (skip for now)
            elif item_type == "reasoning":
                pass

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

        messages, tools, response_format, generation_params = _get_responses_inputs(
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

        messages, tools, response_format, generation_params = _get_responses_inputs(
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
            raise NotImplementedError("ArkLlm does not support streaming call.")
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
