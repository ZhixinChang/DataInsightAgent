from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call, wrap_model_call, ModelRequest, ModelResponse, dynamic_prompt
from langgraph.types import Interrupt
from langchain.messages import HumanMessage, AIMessage, AIMessageChunk, AnyMessage, ToolMessage
from langchain_openai import ChatOpenAI
from typing import TypedDict
from pydantic import BaseModel, Field
from .analytical_tools import data_describe, correlation_analysis


class Context(TypedDict):
    chat_ui: bool

@dynamic_prompt
def chat_ui_system_prompt(request: ModelRequest) -> str:
    """Generate system prompt based on user role."""
    chat_ui = request.runtime.context.get("chat_ui", False)
    base_prompt = '你是一个数据团队的管理者，请结合用户需求使用合适的团队工具解决分析问题，请客观地传达任务信息，不要额外补充用户没提到的需求。当分析完成时请输出工具返回的结果。'

    if chat_ui:
        file_path = '/Users/bytedance/zhangzhixin/GitHub/DataInsightAgent/chat_ui_fastapi/uploaded_files/所有文件'
        result_path = '/Users/bytedance/zhangzhixin/GitHub/DataInsightAgent/chat_ui_fastapi/uploaded_files/分析结果'
        return f"文件所在路径目录：{file_path}，分析结果保存路径目录：{result_path}，{base_prompt}"
    else:
        return base_prompt


def create_supervisor_agent(base_url, api_key, model, tools, checkpointer):
    model_client = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort="low",
        stream_usage=True,
        temperature=0,
        tags=["Supervisor"]
    )


    supervisor_agent = create_agent(name='Supervisor',
                                     model=model_client,
                                     tools=tools,
                                     checkpointer=checkpointer,
                                     system_prompt='你是一个数据团队的管理者，请结合用户需求使用合适的团队工具解决分析问题，请客观地传达任务信息，不要额外补充用户没提到的需求。当分析完成时请输出工具返回的结果。',
                                     middleware=[chat_ui_system_prompt],
                                    context_schema=Context
                                         )
    return supervisor_agent


@wrap_tool_call
def handle_tool_errors(request, handler):
    """Handle tool execution errors with custom messages."""
    try:
        return handler(request)
    except Exception as e:
        # Return a custom error message to the model
        return ToolMessage(
            content=f"Tool error: Please check your input and try again. ({str(e)})",
            tool_call_id=request.tool_call["id"]
        )

class DataAnalystResponseFormat(BaseModel):
    """数据分析师格式化回复"""
    text_result: str = Field(description="文本的处理结果")
    file_result: str = Field(description="文件的保存路径，例如'/.../数据集名称.xlsx'")

def create_data_analyst_agent(base_url, api_key, model):
    model_client = ChatOpenAI(
        base_url=base_url,
        api_key=api_key,
        model=model,
        reasoning_effort="low",
        stream_usage=True,
        temperature=0,
        tags=["Data_analyst"]
    )
    data_analyst_agent = create_agent(name='Data_analyst',
                                      model=model_client,
                                      tools=[data_describe, correlation_analysis],
                                      system_prompt='你是一个数据分析专家，请使用工具解决分析问题，先使用data_describe获取数据集相关信息，再使用其他工具进行分析。当使用工具所需信息不足或工具使用报错时请反馈问题和推理原因，当分析完成时请输出工具返回的结果。',
                                      middleware=[handle_tool_errors],
                                      response_format=DataAnalystResponseFormat
                                      )
    return data_analyst_agent