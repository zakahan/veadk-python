import json
from typing import List, Dict, Optional, Union

from google.adk.models import LlmResponse
from google.adk.models.cache_metadata import CacheMetadata
from google.genai import types
from volcenginesdkarkruntime.types.responses import (
    EasyInputMessageParam,
    ResponseInputTextParam,
    ResponseReasoningItem,
    ResponseOutputMessage,
    ResponseFunctionToolCall,
    ResponseOutputText,
    ResponseTextDeltaEvent,
    ResponseStreamEvent,
    ResponseCompletedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
)
from volcenginesdkarkruntime.types.responses import Response as ArkTypeResponse
from volcenginesdkarkruntime.types.responses.response_input_param import (
    ResponseInputItemParam,
)

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


def build_cache_metadata(response_id: str) -> CacheMetadata:
    """Create a new CacheMetadata instance for agent response tracking.

    Args:
        response_id: Response ID to track

    Returns:
        A new CacheMetadata instance with the agent-response mapping
    """
    if "contents_count" in CacheMetadata.model_fields:  # adk >= 1.17
        cache_metadata = CacheMetadata(
            cache_name=response_id,
            expire_time=0,
            fingerprint="",
            invocations_used=0,
            contents_count=0,
        )
    else:  # 1.15 <= adk < 1.17
        cache_metadata = CacheMetadata(
            cache_name=response_id,
            expire_time=0,
            fingerprint="",
            invocations_used=0,
            cached_contents_count=0,
        )
    return cache_metadata


# ---------------------------------------
# input transfer ------------------------
def get_model_without_provider(request_data: dict) -> dict:
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


def filtered_inputs(
    inputs: List[ResponseInputItemParam],
) -> List[ResponseInputItemParam]:
    # Keep the first message and all consecutive user messages from the end
    # Collect all consecutive user messages from the end
    new_inputs = []
    for m in reversed(inputs):  # Skip the first message
        if m.get("type") == "function_call_output" or m.get("role") == "user":
            new_inputs.append(m)
        else:
            break  # Stop when we encounter a non-user message

    return new_inputs[::-1]


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
    # 1. model provider
    request_data = get_model_without_provider(request_data)

    # 2. filtered input
    request_data["input"] = filtered_inputs(request_data["input"])

    # 3. filter not support data
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
    instructions: Optional[str] = request_data.pop("instructions", None)
    if instructions and not request_data.get("previous_response_id"):
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


# ---------------------------------------
# output transfer -----------------------
def event_to_generate_content_response(
    event: Union[ArkTypeResponse, ResponseStreamEvent],
    *,
    is_partial: bool = False,
    model_version: str = None,
) -> Optional[LlmResponse]:
    parts = []
    if not is_partial:
        for output in event.output:
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

    else:
        if isinstance(event, ResponseReasoningSummaryTextDeltaEvent):
            parts.append(types.Part.from_text(text=event.delta))
        elif isinstance(event, ResponseTextDeltaEvent):
            parts.append(types.Part.from_text(text=event.delta))
        elif isinstance(event, ResponseCompletedEvent):
            raw_response = event.response
            llm_response = ark_response_to_generate_content_response(raw_response)
            return llm_response
        else:
            return None
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

    llm_response = event_to_generate_content_response(
        raw_response, model_version=raw_response.model, is_partial=False
    )
    llm_response.finish_reason = finish_reason
    if raw_response.usage:
        llm_response.usage_metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=raw_response.usage.input_tokens,
            candidates_token_count=raw_response.usage.output_tokens,
            total_token_count=raw_response.usage.total_tokens,
            cached_content_token_count=raw_response.usage.input_tokens_details.cached_tokens,
        )

    # previous_response_id
    previous_response_id = raw_response.id
    llm_response.cache_metadata = build_cache_metadata(previous_response_id)

    return llm_response
