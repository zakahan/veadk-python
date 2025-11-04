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

import asyncio
import sys
from typing import Callable

from veadk.tracing.telemetry.telemetry import (
    trace_call_llm,
    trace_send_data,
    trace_tool_call,
)
from veadk.utils.logger import get_logger

logger = get_logger(__name__)


def patch_asyncio():
    """Patch asyncio to ignore 'Event loop is closed' error.

    After invoking MCPToolset, we met the `RuntimeError: Event loop is closed` error. Related issue see:
    - https://github.com/google/adk-python/issues/1429
    - https://github.com/google/adk-python/pull/1420
    """
    original_del = asyncio.base_subprocess.BaseSubprocessTransport.__del__

    def patched_del(self):
        try:
            original_del(self)
        except RuntimeError as e:
            if "Event loop is closed" not in str(e):
                raise

    asyncio.base_subprocess.BaseSubprocessTransport.__del__ = patched_del

    from anyio._backends._asyncio import CancelScope

    original_cancel_scope_exit = CancelScope.__exit__

    def patched_cancel_scope_exit(self, exc_type, exc_val, exc_tb):
        try:
            return original_cancel_scope_exit(self, exc_type, exc_val, exc_tb)
        except RuntimeError as e:
            if (
                "Attempted to exit cancel scope in a different task than it was entered in"
                in str(e)
            ):
                return False
            raise

    CancelScope.__exit__ = patched_cancel_scope_exit


def patch_google_adk_telemetry() -> None:
    trace_functions = {
        "trace_tool_call": trace_tool_call,
        "trace_call_llm": trace_call_llm,
        "trace_send_data": trace_send_data,
    }

    for mod_name, mod in sys.modules.items():
        if mod_name.startswith("google.adk"):
            for var_name in dir(mod):
                var = getattr(mod, var_name, None)
                if var_name in trace_functions.keys() and isinstance(var, Callable):
                    setattr(mod, var_name, trace_functions[var_name])
                    logger.debug(
                        f"Patch {mod_name} {var_name} with {trace_functions[var_name]}"
                    )


def patch_litellm_responses_handler() -> None:
    """Patch litellm ResponsesToCompletionBridgeHandler.transformation_handler."""
    # Prevent duplicate patches
    if hasattr(patch_litellm_responses_handler, "_patched"):
        logger.debug("ResponsesToCompletionBridgeHandler already patched, skipping")
        return

    try:
        from litellm.completion_extras.litellm_responses_transformation.handler import (
            ResponsesToCompletionBridgeHandler,
            responses_api_bridge,
        )
        from litellm.completion_extras.litellm_responses_transformation.transformation import (
            LiteLLMResponsesTransformationHandler,
        )

        class CustomLiteLLMResponsesTransformationHandler(
            LiteLLMResponsesTransformationHandler
        ):
            """Custom implementation of LiteLLMResponsesTransformationHandler"""

            def __init__(self):
                super().__init__()
                logger.debug("Using custom LiteLLMResponsesTransformationHandler")

            def transform_request(self, *args, **kwargs):
                result = super().transform_request(*args, **kwargs)
                # append custom param for responses api
                previous_responses_id = (
                    kwargs.get("optional_params", {})
                    .get("extra_body", {})
                    .get("previous_response_id")
                )
                if previous_responses_id:
                    result["previous_responses_id"] = previous_responses_id
                return result

            def transform_response(self, *args, **kwargs):
                result = super().transform_response(*args, **kwargs)
                raw_response = kwargs.get("raw_response")
                if raw_response and hasattr(raw_response, "id"):
                    result.id = raw_response.id
                return result

        def patched_init(self):
            super(ResponsesToCompletionBridgeHandler, self).__init__()
            self.transformation_handler = CustomLiteLLMResponsesTransformationHandler()
            logger.debug(
                "Initialized ResponsesToCompletionBridgeHandler with custom transformation_handler"
            )

        ResponsesToCompletionBridgeHandler.__init__ = patched_init

        # Update the existing responses_api_bridge instance
        if hasattr(responses_api_bridge, "transformation_handler"):
            responses_api_bridge.transformation_handler = (
                CustomLiteLLMResponsesTransformationHandler()
            )
            logger.debug(
                "Updated existing responses_api_bridge.transformation_handler with custom implementation"
            )

        # Marked as patched to prevent duplicate application
        patch_litellm_responses_handler._patched = True
        logger.info(
            "Successfully patched ResponsesToCompletionBridgeHandler.__init__ and updated existing instances"
        )

    except ImportError as e:
        logger.warning(f"Failed to patch litellm handler: {e}")


#
# def patch_model_response_to_generate_content_response() -> None:
#     """Patch _model_response_to_generate_content_response to add raw_response.id to custom_metadata."""
#     # Prevent duplicate patches
#     if hasattr(patch_model_response_to_generate_content_response, "_patched"):
#         logger.debug(
#             "_model_response_to_generate_content_response already patched, skipping"
#         )
#         return
#
#     try:
#         from litellm.types.utils import ModelResponse
#         from google.adk.models.lite_llm import (
#             _model_response_to_generate_content_response,
#         )
#
#         original_model_response_to_generate_content_response = (
#             _model_response_to_generate_content_response
#         )
#
#         def patched_model_response_to_generate_content_response(
#             response: ModelResponse,
#         ):
#             """Patched version that adds raw_response.id to custom_metadata."""
#             # Call the original function to get the result
#             llm_response = original_model_response_to_generate_content_response(
#                 response
#             )
#             if not response.id.startswith("chatcmpl"):
#                 if llm_response.custom_metadata is None:
#                     llm_response.custom_metadata = {}
#                 llm_response.custom_metadata["response_id"] = response["id"]
#             return llm_response
#
#         import google.adk.models.lite_llm
#
#         google.adk.models.lite_llm._model_response_to_generate_content_response = (
#             patched_model_response_to_generate_content_response
#         )
#
#         # Prevent duplicate application
#         patch_model_response_to_generate_content_response._patched = True
#         logger.info("Successfully patched _model_response_to_generate_content_response")
#
#     except ImportError as e:
#         logger.warning(
#             f"Failed to patch _model_response_to_generate_content_response: {e}"
#         )
