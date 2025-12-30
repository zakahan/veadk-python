# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# adapted from Google ADK models adk-python/blob/main/src/google/adk/models/lite_llm.py at f1f44675e4a86b75e72cfd838efd8a0399f23e24 · google/adk-python

import base64
import json
from typing import Any, Dict, Union, AsyncGenerator, Tuple, List, Optional, Literal

import openai
from openai.types.responses import Response as OpenAITypeResponse, ResponseStreamEvent
from volcenginesdkarkruntime.types.responses import FunctionToolParam

from volcenginesdkarkruntime.types.responses.response_input_param import (
    ResponseInputItemParam,
    ResponseFunctionToolCallParam,
    EasyInputMessageParam,
    FunctionCallOutput
)
from volcenginesdkarkruntime.types.responses.response_input_message_content_list_param import (
    ResponseInputTextParam,
    ResponseInputImageParam,
    ResponseInputVideoParam,
    ResponseInputAudioParam,
    ResponseInputFileParam,
    ResponseInputContentParam
)


from google.adk.models import LlmRequest, LlmResponse
from google.adk.models.lite_llm import (
    LiteLlm,
    _get_completion_inputs,
    FunctionChunk,
    TextChunk,
    _message_to_generate_content_response,
    UsageMetadataChunk,
)
from google.genai import types
from litellm import ChatCompletionAssistantMessage
from litellm.types.utils import (
    ChatCompletionMessageToolCall,
    Function,
)
from pydantic import Field
from volcenginesdkarkruntime import AsyncArk

from veadk.config import settings
from veadk.consts import DEFAULT_VIDEO_MODEL_API_BASE
from veadk.models.ark_transform import (
    CompletionToResponsesAPIHandler,
)
from veadk.utils.logger import get_logger

logger = get_logger(__name__)


def _to_responses_api_role(role: Optional[str]) -> Literal["user", "assistant"]:
    if role in ["model", "assistant"]:
        return "assistant"
    return "user"


def _safe_json_serialize(obj) -> str:  # fixme: 这个可能得修改，可能要扁平化处理
    """Convert any Python object to a JSON-serializable type or string.

    Args:
      obj: The object to serialize.

    Returns:
      The JSON-serialized object string or string.
    """

    try:
        # Try direct JSON serialization first
        return json.dumps(obj, ensure_ascii=False)
    except (TypeError, OverflowError):
        return str(obj)


def _file_data_to_content_param(
    part: types.Part,
) -> ResponseInputContentParam:
    file_uri = part.file_data.file_uri
    mime_type = part.file_data.mime_type
    display_name = part.file_data.display_name
    fps = 1.0
    if getattr(part, "video_metadata", None):
        video_metadata = part.video_metadata
        if isinstance(video_metadata, dict):
            fps = video_metadata.get("fps")
        else:
            fps = getattr(video_metadata, "fps", 1)

    is_file_id = file_uri.startswith("file_id:")
    value = file_uri[7:] if is_file_id else file_uri
    # video
    if mime_type.startswith("video/"):
        param = {"file_id": value} if is_file_id else {"video_url": value}
        if fps is not None:
            param["fps"] = fps
        return ResponseInputVideoParam(
            type="input_video",
            **param,
        )
    # image
    if mime_type.startswith("image/"):
        return ResponseInputImageParam(
            type="input_image",
            detail="auto",
            **({"file_id": value} if is_file_id else {"image_url": value}),
        )
    # file
    param = {"file_id": value} if is_file_id else {"file_url": value}
    if display_name:
        param['filename'] = display_name
    return ResponseInputFileParam(
        type="input_file",
        **param,
    )


def _inline_data_to_content_param(part: types.Part) -> ResponseInputContentParam:
    mime_type = (part.inline_data.mime_type if part.inline_data else None) or "application/octet-stream"
    base64_string = base64.b64encode(part.inline_data.data).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{base64_string}"

    if mime_type.startswith("image"):
        return ResponseInputImageParam(
            type="input_image",
            image_url=data_uri,
            detail="auto",
        )
    if mime_type.startswith("video"):
        param: Dict[str, Any] = {"video_url": data_uri}
        if getattr(part, "video_metadata", None):
            video_metadata = part.video_metadata
            if isinstance(video_metadata, dict):
                fps = video_metadata.get("fps")
            else:
                fps = getattr(video_metadata, "fps", None)
            if fps is not None:
                param["fps"] = fps
        return ResponseInputVideoParam(
            type="input_video",
            **param,
        )

    file_param: Dict[str, Any] = {"file_data": data_uri}
    if getattr(part.inline_data, "display_name", None):
        file_param["filename"] = part.inline_data.display_name
    return ResponseInputFileParam(
        type="input_file",
        **file_param,
    )


def _get_content(
        parts: List[types.Part],
        role: Literal["user", "system", "developer", "assistant"],
) -> Optional[EasyInputMessageParam]:
    content = []
    for part in parts:
        if part.text:
            content.append(
                ResponseInputTextParam(
                    type="input_text",
                    text=part.text,
                )
            )
        elif part.inline_data and part.inline_data.data:
            content.append(_inline_data_to_content_param(part))
        elif part.file_data:  # 有两种，file_id和file_url
            content.append(_file_data_to_content_param(part))
    if len(content)>0:
        return EasyInputMessageParam(
            type="message",
            role=role,
            content=content
        )
    else:
        return None

def _content_to_input_item(
    content: types.Content,
) -> Union[ResponseInputItemParam, List[ResponseInputItemParam]]:
    role = _to_responses_api_role(content.role)

    # 1. FunctionResponse：单独坐一桌，Tool消息不能混合其他内容，收集后直接返回
    input_list = []
    for part in content.parts:
        if part.function_response:  # FunctionCallOutput
            input_list.append(
                FunctionCallOutput(
                    call_id=part.function_response.id,
                    output=_safe_json_serialize(part.function_response.response),
                    type="function_call_output",
                )
            )
    if input_list:
        return input_list if len(input_list) > 1 else input_list[0]

    input_content = _get_content(content.parts, role=role) or None

    if role == "user":
        # 2. 处理user的消息
        # user_content 只可能是一条
        if input_content:
            return input_content
    else:  # model
        # 3. 处理model的消息
        content_present = False
        for part in content.parts:
            if part.function_call:
                input_list.append(
                    ResponseFunctionToolCallParam(
                        arguments=_safe_json_serialize(part.function_call.args),
                        call_id=part.function_call.id,
                        name=part.function_call.name,
                        type="function_call",
                    )
                )
            # 我怀疑这种输入param里同时有text和function_call的情况绝对很少了
            elif part.text or part.inline_data:
                if input_content:
                    input_list.append(input_content)
    return input_list


def _function_declarations_to_tool_param(
        function_declaration: types.FunctionDeclaration
) -> FunctionToolParam:
    assert function_declaration.name
    tool_params = FunctionToolParam(
        name=function_declaration.name,
        parameters=...,     # todo:here
        type="function",
        description=function_declaration.description,
    )



async def _get_responses_inputs(
    llm_request: LlmRequest,
    model: str,
) -> Tuple:
    inputs: List[ResponseInputItemParam] = []
    for content in llm_request.contents or []:
        # 每个content，代表`一次对话`，这`一次对话`可能有`多个内容`，但不可能有`多个对话`
        input_item_or_list = _content_to_input_item(content)
        if isinstance(input_item_or_list,list):
            inputs.extend(input_item_or_list)
        elif input_item_or_list:
            inputs.append(input_item_or_list)

    # 将system_prompt插入到开头
    if llm_request.config.system_instruction:
        inputs.insert(
            0,
            EasyInputMessageParam(
                role="system",
                type="message",
                content=[ResponseInputTextParam(
                    type="input_text",
                    text=llm_request.config.system_instruction,
                )]
            )
        )

    # 2. Convert tool declarations
    tools: Optional[List[FunctionToolParam]] = None
    if (
        llm_request.config
        and llm_request.config.tools
        and llm_request.config.tools[0].function_declarations
    ):
        tools = [
            _function_declarations_to_tool_param(tool)
            for tool in llm_request.config.tools[0].function_declarations
        ]



class ArkLlmClient:
    async def aresponse(
        self, **kwargs
    ) -> Union[OpenAITypeResponse, openai.AsyncStream[ResponseStreamEvent]]:
        # 1. Get request params
        api_base = kwargs.pop("api_base", DEFAULT_VIDEO_MODEL_API_BASE)
        api_key = kwargs.pop("api_key", settings.model.api_key)

        # 2. Call openai responses
        client = AsyncArk(
            base_url=api_base,
            api_key=api_key,
        )

        raw_response = await client.responses.create(**kwargs)
        return raw_response


class ArkLlm(LiteLlm):
    llm_client: ArkLlmClient = Field(default_factory=ArkLlmClient)
    _additional_args: Dict[str, Any] = None
    transform_handler: CompletionToResponsesAPIHandler = Field(
        default_factory=CompletionToResponsesAPIHandler
    )

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
        # logger.debug(_build_request_log(llm_request))

        messages, tools, response_format, generation_params = _get_completion_inputs(
            llm_request
        )

        if "functions" in self._additional_args:
            # LiteLLM does not support both tools and functions together.
            tools = None
        # ------------------------------------------------------ #
        # get previous_response_id
        previous_response_id = None
        if llm_request.cache_metadata and llm_request.cache_metadata.cache_name:
            previous_response_id = llm_request.cache_metadata.cache_name
        completion_args = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "response_format": response_format,
            "previous_response_id": previous_response_id,  # supply previous_response_id
        }
        # ------------------------------------------------------ #
        completion_args.update(self._additional_args)

        if generation_params:
            completion_args.update(generation_params)
        response_args = self.transform_handler.transform_request(**completion_args)

        if stream:
            text = ""
            # Track function calls by index
            function_calls = {}  # index -> {name, args, id}
            response_args["stream"] = True
            aggregated_llm_response = None
            aggregated_llm_response_with_tool_call = None
            usage_metadata = None
            fallback_index = 0
            raw_response = await self.llm_client.aresponse(**response_args)
            async for part in raw_response:
                for (
                    model_response,
                    chunk,
                    finish_reason,
                ) in self.transform_handler.stream_event_to_chunk(
                    part, model=self.model
                ):
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
                        # ------------------------------------------------------ #
                        if model_response.get("usage", {}).get("prompt_tokens_details"):
                            usage_metadata.cached_content_token_count = (
                                model_response.get("usage", {})
                                .get("prompt_tokens_details")
                                .cached_tokens
                            )
                        # ------------------------------------------------------ #

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
                        self.transform_handler.adapt_responses_api(
                            model_response,
                            aggregated_llm_response_with_tool_call,
                            stream=True,
                        )
                        text = ""
                        function_calls.clear()
                    elif finish_reason == "stop" and text:
                        aggregated_llm_response = _message_to_generate_content_response(
                            ChatCompletionAssistantMessage(
                                role="assistant", content=text
                            )
                        )
                        self.transform_handler.adapt_responses_api(
                            model_response,
                            aggregated_llm_response,
                            stream=True,
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
            raw_response = await self.llm_client.aresponse(**response_args)
            for (
                llm_response
            ) in self.transform_handler.openai_response_to_generate_content_response(
                llm_request, raw_response
            ):
                yield llm_response
