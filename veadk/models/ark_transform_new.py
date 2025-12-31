import json
from typing import List, Dict

from google.adk.models import LlmResponse
from google.genai import types
from volcenginesdkarkruntime.types.responses import (
    EasyInputMessageParam,
    ResponseInputTextParam,
    ResponseOutputItem,
    ResponseReasoningItem,
    ResponseOutputMessage,
    ResponseFunctionToolCall,
    ResponseOutputText,
)
from volcenginesdkarkruntime.types.responses import Response as ArkTypeResponse

from veadk.utils.logger import get_logger

logger = get_logger(__name__)


_FINISH_REASON_MAPPING = {
    "incomplete": {
        "length": types.FinishReason.MAX_TOKENS,
        "content_filter": types.FinishReason.SAFETY,
    },
    "completed": {
        "other": types.FinishReason.STOP,
    },
}

ark_supported_fields = [
    "input",
    "model",
    "stream",
    "background",
    "include",
    "instructions",
    "max_output_tokens",
    "parallel_tool_calls",
    "previous_response_id",
    "thinking",
    "store",
    "caching",
    "stream",
    "temperature",
    "text",
    "tool_choice",
    "tools",
    "top_p",
    "max_tool_calls",
    "expire_at",
    "extra_headers",
    "extra_query",
    "extra_body",
    "timeout",
    "reasoning"
    # auth params
    "api_key",
    "api_base",
]


def openai_field_reorganization(request_data: dict) -> dict:
    model = request_data.get("model")

    if not isinstance(model, str):
        raise ValueError(
            "Unsupported Responses API request: 'model' must be a string in the OpenAI-style format, e.g. 'openai/gpt-4o'."
        )

    if "/" not in model:
        raise ValueError(
            "Unsupported Responses API request: only OpenAI-style model names are supported (use 'openai/<model>')."
        )

    provider, actual_model = model.split("/", 1)
    if provider != "openai":
        raise ValueError(
            f"Unsupported model prefix '{provider}'. Responses API request format only supports 'openai/<model>'."
        )

    request_data["model"] = actual_model

    return request_data


def _is_caching_enabled(request_data: dict) -> bool:
    extra_body = request_data.get("extra_body")
    if not isinstance(extra_body, dict):
        return False
    caching = extra_body.get("caching")
    if not isinstance(caching, dict):
        return False
    return caching.get("type") == "enabled"


def _remove_caching(request_data: dict) -> None:
    extra_body = request_data.get("extra_body")
    if isinstance(extra_body, dict):
        extra_body.pop("caching", None)
    request_data.pop("caching", None)


def request_reorganization_by_ark(request_data: Dict) -> Dict:
    # model provider
    request_data = openai_field_reorganization(request_data)

    # filter not support data
    request_data = {
        key: value for key, value in request_data.items() if key in ark_supported_fields
    }

    if _is_caching_enabled(request_data) and request_data.get("text") is not None:
        logger.warning(
            "Caching is enabled, but text is provided. Ark does not support caching with text. Caching will be disabled."
        )
        _remove_caching(request_data)

    # [Note: Ark Limitations] tools and previous_response_id
    # Remove tools in subsequent rounds (when previous_response_id is present)
    if (
        "tools" in request_data
        and "previous_response_id" in request_data
        and request_data["previous_response_id"] is not None
    ):
        # Remove tools in subsequent rounds regardless of caching status
        del request_data["tools"]

    # [Note: Ark Limitations] caching and store
    # Ensure store field is true or default when caching is enabled
    if _is_caching_enabled(request_data):
        # Set store to true when caching is enabled for writing
        if "store" not in request_data:
            request_data["store"] = True
        elif request_data["store"] is False:
            # Override false to true for cache writing
            request_data["store"] = True

    # [NOTE Ark Limitations] instructions -> input (because of caching)
    # Due to the Volcano Ark settings, there is a conflict between the cache and the instructions field.
    # If a system prompt is needed, it should be placed in the system role message within the input, instead of using the instructions parameter.
    # https://www.volcengine.com/docs/82379/1585128
    instructions = request_data.pop("instructions", None)
    if instructions and isinstance(instructions, str):
        request_data["input"].insert(
            0,
            EasyInputMessageParam(
                role="system",
                type="message",
                content=[
                    ResponseInputTextParam(
                        type="input_text",
                        text=instructions,
                    )
                ],
            ),
        )

    return request_data


def _output_to_generate_content_response(
    outputs: List[ResponseOutputItem],
    *,
    is_partial: bool = False,
    model_version: str = None,
):
    parts = []
    for output in outputs:
        if isinstance(output, ResponseReasoningItem):
            parts.append(
                types.Part(
                    text="\n".join([summary.text for summary in output.summary]),
                    thought=True,
                )
            )
        elif isinstance(output, ResponseOutputMessage):
            text = ""
            if isinstance(output.content, list):
                for item in output.content:
                    if isinstance(item, ResponseOutputText):
                        text += item.text
            parts.append(types.Part(text=text))

        elif isinstance(output, ResponseFunctionToolCall):
            part = types.Part.from_function_call(
                name=output.name, args=json.loads(output.arguments or "{}")
            )
            part.function_call.id = output.call_id
            parts.append(part)

    return LlmResponse(
        content=types.Content(role="model", parts=parts),
        partial=is_partial,
        model_version=model_version,
    )


def ark_response_to_generate_content_response(
    raw_response: ArkTypeResponse,
) -> LlmResponse:
    """
    ArkTypeResponse -> LlmResponse
    instead of `_model_response_to_generate_content_response`,
    """
    outputs = raw_response.output
    status = raw_response.status
    incomplete_details = getattr(
        raw_response.incomplete_details or None, "reason", "other"
    )

    finish_reason = _FINISH_REASON_MAPPING.get(status, {}).get(
        incomplete_details, types.FinishReason.OTHER
    )

    if not outputs:
        raise ValueError("No message in response")

    llm_response = _output_to_generate_content_response(
        outputs, model_version=raw_response.model
    )
    llm_response.finish_reason = finish_reason
    if raw_response.usage:
        llm_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=raw_response.usage.input_tokens,
            candidates_token_count=raw_response.usage.output_tokens,
            total_token_count=raw_response.usage.total_tokens,
            cached_content_token_count=raw_response.usage.input_tokens_details.cached_tokens,
        )

    return llm_response
