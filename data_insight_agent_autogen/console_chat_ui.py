import asyncio
import os
import sys
import time
from inspect import iscoroutinefunction
from typing import AsyncGenerator, Awaitable, Callable, Dict, List, Optional, TypeVar, Union, cast

from autogen_core import CancellationToken
from autogen_core.models import RequestUsage

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.base import Response, TaskResult
from autogen_agentchat.messages import (
    BaseAgentEvent,
    BaseChatMessage,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    UserInputRequestedEvent,
TextMessage, ToolCallRequestEvent, ToolCallExecutionEvent, ToolCallSummaryMessage, ThoughtEvent
)
import re

def _is_running_in_iterm() -> bool:
    return os.getenv("TERM_PROGRAM") == "iTerm.app"


def _is_output_a_tty() -> bool:
    return sys.stdout.isatty()


SyncInputFunc = Callable[[str], str]
AsyncInputFunc = Callable[[str, Optional[CancellationToken]], Awaitable[str]]
InputFuncType = Union[SyncInputFunc, AsyncInputFunc]

T = TypeVar("T", bound=TaskResult | Response)



class UserInputManager:
    def __init__(self, callback: InputFuncType):
        self.input_events: Dict[str, asyncio.Event] = {}
        self.callback = callback


    def get_wrapped_callback(self) -> AsyncInputFunc:
        async def user_input_func_wrapper(prompt: str, cancellation_token: Optional[CancellationToken]) -> str:
            # Lookup the event for the prompt, if it exists wait for it.
            # If it doesn't exist, create it and store it.
            # Get request ID:
            request_id = UserProxyAgent.InputRequestContext.request_id()
            if request_id in self.input_events:
                event = self.input_events[request_id]
            else:
                event = asyncio.Event()
                self.input_events[request_id] = event

            await event.wait()

            del self.input_events[request_id]

            if iscoroutinefunction(self.callback):
                # Cast to AsyncInputFunc for proper typing
                async_func = cast(AsyncInputFunc, self.callback)
                return await async_func(prompt, cancellation_token)
            else:
                # Cast to SyncInputFunc for proper typing
                sync_func = cast(SyncInputFunc, self.callback)
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, sync_func, prompt)

        return user_input_func_wrapper



    def notify_event_received(self, request_id: str) -> None:
        if request_id in self.input_events:
            self.input_events[request_id].set()
        else:
            event = asyncio.Event()
            self.input_events[request_id] = event





async def console_chat_ui(
    stream: AsyncGenerator[BaseAgentEvent | BaseChatMessage | T, None],
    *,
    no_inline_images: bool = False,
    output_stats: bool = False,
    user_input_manager: UserInputManager | None = None,
):
    """
    Consumes the message stream from :meth:`~autogen_agentchat.base.TaskRunner.run_stream`
    or :meth:`~autogen_agentchat.base.ChatAgent.on_messages_stream` and renders the messages to the console.
    Returns the last processed TaskResult or Response.

    .. note::

        `output_stats` is experimental and the stats may not be accurate.
        It will be improved in future releases.

    Args:
        stream (AsyncGenerator[BaseAgentEvent | BaseChatMessage | TaskResult, None] | AsyncGenerator[BaseAgentEvent | BaseChatMessage | Response, None]): Message stream to render.
            This can be from :meth:`~autogen_agentchat.base.TaskRunner.run_stream` or :meth:`~autogen_agentchat.base.ChatAgent.on_messages_stream`.
        no_inline_images (bool, optional): If terminal is iTerm2 will render images inline. Use this to disable this behavior. Defaults to False.
        output_stats (bool, optional): (Experimental) If True, will output a summary of the messages and inline token usage info. Defaults to False.

    Returns:
        last_processed: A :class:`~autogen_agentchat.base.TaskResult` if the stream is from :meth:`~autogen_agentchat.base.TaskRunner.run_stream`
            or a :class:`~autogen_agentchat.base.Response` if the stream is from :meth:`~autogen_agentchat.base.ChatAgent.on_messages_stream`.
    """
    render_image_iterm = _is_running_in_iterm() and _is_output_a_tty() and not no_inline_images
    start_time = time.time()
    total_usage = RequestUsage(prompt_tokens=0, completion_tokens=0)

    last_processed: Optional[T] = None

    streaming_chunks: List[str] = []

    async for message in stream:
        if isinstance(message, TaskResult):
            duration = time.time() - start_time
            if output_stats:
                output = (
                    f"{'-' * 10} Summary {'-' * 10}<br>"
                    f"Number of messages: {len(message.messages)}<br>"
                    f"Finish reason: {message.stop_reason}<br>"
                    f"Total prompt tokens: {total_usage.prompt_tokens}<br>"
                    f"Total completion tokens: {total_usage.completion_tokens}<br>"
                    f"Duration: {duration:.2f} seconds<br>"
                )
                yield output

            # mypy ignore
            last_processed = message  # type: ignore

        elif isinstance(message, Response):
            duration = time.time() - start_time

            # Print final response.
            if isinstance(message.chat_message, MultiModalMessage):
                final_content = message.chat_message.to_text(iterm=render_image_iterm)
            else:
                final_content = message.chat_message.to_text()
            output = f"{'-' * 10} {message.chat_message.source} {'-' * 10}<br>{final_content}<br>"
            if message.chat_message.models_usage:
                if output_stats:
                    output += f"[Prompt tokens: {message.chat_message.models_usage.prompt_tokens}, Completion tokens: {message.chat_message.models_usage.completion_tokens}]\n"
                total_usage.completion_tokens += message.chat_message.models_usage.completion_tokens
                total_usage.prompt_tokens += message.chat_message.models_usage.prompt_tokens
            yield output

            # Print summary.
            if output_stats:
                if message.inner_messages is not None:
                    num_inner_messages = len(message.inner_messages)
                else:
                    num_inner_messages = 0
                output = (
                    f"{'-' * 10} Summary {'-' * 10}<br>"
                    f"Number of inner messages: {num_inner_messages}<br>"
                    f"Total prompt tokens: {total_usage.prompt_tokens}<br>"
                    f"Total completion tokens: {total_usage.completion_tokens}<br>"
                    f"Duration: {duration:.2f} seconds<br>"
                )
                yield output

            # mypy ignore
            last_processed = message  # type: ignore
        # We don't want to print UserInputRequestedEvent messages, we just use them to signal the user input event.
        elif isinstance(message, UserInputRequestedEvent):
            if user_input_manager is not None:
                user_input_manager.notify_event_received(message.request_id)
        else:
            # Cast required for mypy to be happy
            message = cast(BaseAgentEvent | BaseChatMessage, message)  # type: ignore
            if not streaming_chunks:
                # Print message sender.
                if isinstance(message, TextMessage):
                    if message.source != 'user':
                        yield f"{'-' * 10} ({message.source}) 回答{'-' * 10}" + '<br>'
                if isinstance(message, ToolCallRequestEvent):
                    yield f"{'-' * 10} ({message.source}) 开始处理需求{'-' * 10}" + '<br>'
                if isinstance(message, ToolCallExecutionEvent):
                    yield f"{'-' * 10} ({message.source}) 需求处理中{'-' * 10}" + '<br>'
                if isinstance(message, ToolCallSummaryMessage):
                    yield f"{'-' * 10} ({message.source}) 需求处理完成{'-' * 10}" + '<br>'
                if isinstance(message, ThoughtEvent):
                    yield f"{'-' * 10} ({message.source}) 思考{'-' * 10}" + '<br>'

            if isinstance(message, ModelClientStreamingChunkEvent):
                yield message.to_text()
                streaming_chunks.append(message.content)
            else:
                if streaming_chunks:
                    streaming_chunks.clear()
                    # Chunked messages are already printed, so we just print a newline.
                    yield "" + '<br>'

                elif isinstance(message, MultiModalMessage):
                    yield message.to_text(iterm=render_image_iterm) + '<br>'
                else:
                    if isinstance(message, TextMessage):
                        if message.source != 'user':
                            yield message.to_text() + '<br>'
                    elif isinstance(message, ToolCallRequestEvent):
                        yield f"({message.source}) ：分配需求给{message.content[0].name}来处理" + '<br>'
                    elif isinstance(message, ToolCallExecutionEvent):
                        yield re.findall(r'IMAGE_PATHS<.*>IMAGE_PATHS|FILE_PATHS<.*>FILE_PATHS', message.to_text())
                        yield f"({message.source}) ：借助{message.content[0].name}完成需求处理" + '<br>'
                    elif isinstance(message, ToolCallSummaryMessage):
                        yield f"({message.source}) 总结处理结果：{message.to_text()}" + '<br>'
                    elif isinstance(message, ThoughtEvent):
                        yield f"({message.source}) 思考：{message.to_text()}" + '<br>'
                    else:
                        yield message.to_text() + '<br>'


                if message.models_usage:
                    if output_stats:
                        yield f"[Prompt tokens: {message.models_usage.prompt_tokens}, Completion tokens: {message.models_usage.completion_tokens}]" + '<br>'

                    total_usage.completion_tokens += message.models_usage.completion_tokens
                    total_usage.prompt_tokens += message.models_usage.prompt_tokens

    if last_processed is None:
        raise ValueError("No TaskResult or Response was processed.")

    # yield last_processed
