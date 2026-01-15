from langchain.messages import HumanMessage, AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, Interrupt

from .sub_agent import create_supervisor_agent, create_data_analyst_agent




class DataInsightTeam:
    def __init__(self, base_url, api_key, model):
        self.data_analyst_agent = create_data_analyst_agent(base_url, api_key, model)

        @tool
        def data_analysis_team(task: str) -> str:
            """数据分析团队工具，负责完成指标相关性的分析，task信息至少需要包含数据集路径、结果保存路径和分析需求的信息"""

            result = self.data_analyst_agent.invoke({"messages": [{"role": "user", "content": task}]})
            return result["messages"][-1].content


        self.supervisor_agent = create_supervisor_agent(base_url, api_key, model, tools=[data_analysis_team],
                                                        checkpointer=InMemorySaver())

    async def astream(self, content='', resume=None):

        config = {"configurable": {"thread_id": "1"}}

        if resume:
            stream_input = Command(resume=resume)
        else:
            stream_input = {"messages": [HumanMessage(content=content)]}

        interrupts = []
        current_agent = None
        async for _, stream_mode, data in self.supervisor_agent.astream(input=stream_input,
                                                                        context={"chat_ui": False}, config=config,
                                                                        stream_mode=["messages", "updates"],
                                                                        subgraphs=True):
            if stream_mode == "messages":
                token, metadata = data
                if tags := metadata.get("tags", []):
                    this_agent = tags[0]
                    if this_agent != current_agent:
                        print(f"{'=' * 34}🤖 {this_agent}{'=' * 34}")
                        current_agent = this_agent
                if isinstance(token, AIMessageChunk) and current_agent == 'Supervisor':
                    self._render_message_chunk(token)
            if stream_mode == "updates":
                for source, update in data.items():
                    if source in ("model", "tools"):
                        self._render_completed_message(update["messages"][-1])
                    if source == "__interrupt__":
                        interrupts.extend(update)
                        self._render_interrupt(update[0])

    async def astream_chat_ui(self, content='', resume=None, thread_id=None):

        config = {"configurable": {"thread_id": thread_id}}

        if resume:
            stream_input = Command(resume=resume)
        else:
            stream_input = {"messages": [HumanMessage(content=content)]}

        interrupts = []
        current_agent = None
        async for _, stream_mode, data in self.supervisor_agent.astream(input=stream_input,
                                                                        context={"chat_ui": True},config=config,
                                                                        stream_mode=["messages", "updates"],
                                                                        subgraphs=True):
            if stream_mode == "messages":
                token, metadata = data
                if tags := metadata.get("tags", []):
                    this_agent = tags[0]
                    if this_agent != current_agent:
                        print(f"{'=' * 34}🤖 {this_agent}{'=' * 34}")
                        current_agent = this_agent
                if isinstance(token, AIMessageChunk) and current_agent == 'Supervisor':
                    if token.text:
                        print(token.text, end="")
                        yield token.text
            if stream_mode == "updates":
                for source, update in data.items():
                    if source in ("model", "tools"):
                        message = update["messages"][-1]
                        if isinstance(message, AIMessage) and message.tool_calls:
                            message.pretty_print()
                            yield f"Tool calls: {message.tool_calls}"

                        if isinstance(message, ToolMessage):
                            message.pretty_print()
                            yield f"Tool response: {message.content_blocks}"

                    if source == "__interrupt__":
                        interrupts.extend(update)
                        interrupt = update[0]
                        interrupts = interrupt.value
                        for request in interrupts["action_requests"]:
                            yield request["description"]

    def _render_message_chunk(self, token: AIMessageChunk) -> None:
        if token.text:
            print(token.text, end="")
        if token.tool_call_chunks:
            # print(token.tool_call_chunks)
            pass

    def _render_completed_message(self, message: AnyMessage) -> None:
        if isinstance(message, AIMessage) and message.tool_calls:
            # print(f"Tool calls: {message.tool_calls}")
            message.pretty_print()
        if isinstance(message, ToolMessage):
            # print(f"Tool response: {message.content_blocks}")
            message.pretty_print()

    def _render_interrupt(self, interrupt: Interrupt) -> None:
        interrupts = interrupt.value
        for request in interrupts["action_requests"]:
            print(request["description"])
