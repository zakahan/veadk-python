from typing import Union

import litellm
from litellm import acompletion, completion
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper

from litellm.types.utils import ModelResponse
from veadk.utils.logger import get_logger


# This will add functions to prompts if functions are provided.
litellm.add_function_to_prompt = True

logger = get_logger(__name__)


class ArkLitellmClient:
    async def acompletion(
        self, model, messages, tools, **kwargs
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """Asynchronously calls acompletion.

        Args:
          model: The model name.
          messages: The messages to send to the model.
          tools: The tools to use for the model.
          **kwargs: Additional arguments to pass to acompletion.

        Returns:
          The model response as a message.
        """

        return await acompletion(
            model=model,
            messages=messages,
            tools=tools,
            **kwargs,
        )

    def completion(
        self, model, messages, tools, stream=False, **kwargs
    ) -> Union[ModelResponse, CustomStreamWrapper]:
        """Synchronously calls completion. This is used for streaming only.

        Args:
          model: The model to use.
          messages: The messages to send.
          tools: The tools to use for the model.
          stream: Whether to stream the response.
          **kwargs: Additional arguments to pass to completion.

        Returns:
          The response from the model.
        """

        return completion(
            model=model,
            messages=messages,
            tools=tools,
            stream=stream,
            **kwargs,
        )
