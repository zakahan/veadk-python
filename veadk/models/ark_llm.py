import json
from typing import Any, Dict, AsyncGenerator, Union

from google.adk.models import LlmRequest, LlmResponse
from litellm import ChatCompletionAssistantMessage

from google.genai import types
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.utils import ModelResponse, ChatCompletionMessageToolCall, Function
from pydantic import Field
import litellm
from google.adk.models.lite_llm import (
    LiteLlm,
    LiteLLMClient,
    _model_response_to_generate_content_response,
    _message_to_generate_content_response,
    TextChunk,
    UsageMetadataChunk,
    _build_request_log,
    _get_completion_inputs,
    _model_response_to_chunk,
    FunctionChunk,
)
from veadk.utils.logger import get_logger


# This will add functions to prompts if functions are provided.
litellm.add_function_to_prompt = True

logger = get_logger(__name__)


class ArkLiteLLMClient(LiteLLMClient):
    def __init__(self):
        super().__init__()

    async def acompletion(
        self, model, messages, tools, **kwargs
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        return await super().acompletion(model, messages, tools, **kwargs)


def _model_response_to_generate_content_response_for_responses_api_handler(
    response: ModelResponse,
):
    """Patched version that adds raw_response.id to custom_metadata."""
    # Call the original function to get the result
    llm_response = _model_response_to_generate_content_response(response)
    if not response.id.startswith("chatcmpl"):
        if llm_response.custom_metadata is None:
            llm_response.custom_metadata = {}
        llm_response.custom_metadata["response_id"] = response["id"]
    return llm_response


# def get_response


class ArkLiteLlm(LiteLlm):
    llm_client: ArkLiteLLMClient = Field(default_factory=ArkLiteLLMClient)
    """The LLM client to use for the model."""

    _additional_args: Dict[str, Any] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
        # 获取previous_response_id

        if "functions" in self._additional_args:
            # LiteLLM does not support both tools and functions together.
            tools = None

        completion_args = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
            "previous_response_id": llm_request.config.labels.get(
                "previous_response_id", None
            ),
        }
        completion_args.update(self._additional_args)

        if generation_params:
            completion_args.update(generation_params)

        if stream:
            text = ""
            # Track function calls by index
            function_calls = {}  # index -> {name, args, id}
            completion_args["stream"] = True
            aggregated_llm_response = None
            aggregated_llm_response_with_tool_call = None
            usage_metadata = None
            fallback_index = 0
            async for part in await self.llm_client.acompletion(**completion_args):
                for chunk, finish_reason in _model_response_to_chunk(part):
                    if isinstance(chunk, FunctionChunk):
                        index = chunk.index or fallback_index
                        if index not in function_calls:
                            function_calls[index] = {"name": "", "args": "", "id": None}

                        if chunk.name:
                            function_calls[index]["name"] += chunk.name
                        if chunk.args:
                            function_calls[index]["args"] += chunk.args

                            # check if args is completed (workaround for improper chunk
                            # indexing)
                            try:
                                json.loads(function_calls[index]["args"])
                                fallback_index += 1
                            except json.JSONDecodeError:
                                pass

                        function_calls[index]["id"] = (
                            chunk.id or function_calls[index]["id"] or str(index)
                        )
                    elif isinstance(chunk, TextChunk):
                        text += chunk.text
                        yield _message_to_generate_content_response(
                            ChatCompletionAssistantMessage(
                                role="assistant",
                                content=chunk.text,
                            ),
                            is_partial=True,
                        )
                    elif isinstance(chunk, UsageMetadataChunk):
                        usage_metadata = types.GenerateContentResponseUsageMetadata(
                            prompt_token_count=chunk.prompt_tokens,
                            candidates_token_count=chunk.completion_tokens,
                            total_token_count=chunk.total_tokens,
                        )

                    if (
                        finish_reason == "tool_calls" or finish_reason == "stop"
                    ) and function_calls:
                        tool_calls = []
                        for index, func_data in function_calls.items():
                            if func_data["id"]:
                                tool_calls.append(
                                    ChatCompletionMessageToolCall(
                                        type="function",
                                        id=func_data["id"],
                                        function=Function(
                                            name=func_data["name"],
                                            arguments=func_data["args"],
                                            index=index,
                                        ),
                                    )
                                )
                        aggregated_llm_response_with_tool_call = (
                            _message_to_generate_content_response(
                                ChatCompletionAssistantMessage(
                                    role="assistant",
                                    content=text,
                                    tool_calls=tool_calls,
                                )
                            )
                        )
                        text = ""
                        function_calls.clear()
                    elif finish_reason == "stop" and text:
                        aggregated_llm_response = _message_to_generate_content_response(
                            ChatCompletionAssistantMessage(
                                role="assistant", content=text
                            )
                        )
                        text = ""

            # waiting until streaming ends to yield the llm_response as litellm tends
            # to send chunk that contains usage_metadata after the chunk with
            # finish_reason set to tool_calls or stop.
            if aggregated_llm_response:
                if usage_metadata:
                    aggregated_llm_response.usage_metadata = usage_metadata
                    usage_metadata = None
                yield aggregated_llm_response

            if aggregated_llm_response_with_tool_call:
                if usage_metadata:
                    aggregated_llm_response_with_tool_call.usage_metadata = (
                        usage_metadata
                    )
                yield aggregated_llm_response_with_tool_call

        else:
            response = await self.llm_client.acompletion(**completion_args)
            yield _model_response_to_generate_content_response_for_responses_api_handler(
                response
            )
