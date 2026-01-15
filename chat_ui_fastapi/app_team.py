import asyncio
import io
import logging
import os
import sys
import uuid
import pandas as pd
from contextlib import asynccontextmanager
from typing import Any, List

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi import UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from uvicorn.config import LOGGING_CONFIG


def add_current_dir_to_path(current_dir):
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        print(f"已将当前目录添加到环境变量")
    else:
        print(f"当前目录已在环境变量中")


add_current_dir_to_path('')

from data_insight_agent import DataInsightTeam

base_url = ""
api_key = ""
model = ""


# -------------------------- 1. 初始化配置（解耦全局变量） --------------------------
async def get_team():
    """Get the assistant agent, load state from file."""
    data_insight_team = DataInsightTeam(base_url=base_url, api_key=api_key, model=model)
    return data_insight_team


class AgentManager:
    """Agent管理类，替代全局变量，降低耦合"""

    def __init__(self):
        self.data_insight_team = None
        self.conversation_map = {}  # 存储对话ID和对应的thread_id映射

    async def get_or_create_conversation(self, conversation_id: str = None):
        """获取或创建对话"""
        if conversation_id:
            if not isinstance(conversation_id, str):
                raise ValueError("对话ID必须是字符串类型")

            if conversation_id.strip() == "":
                raise ValueError("对话ID不能为空字符串")

            if conversation_id in self.conversation_map:
                # 如果对话ID存在，返回对应的thread_id
                return self.conversation_map[conversation_id]

        # 如果对话ID不存在或未提供，创建新的对话
        if not conversation_id:
            conversation_id = str(uuid.uuid4())

        # 将conversationId+user的字符串直接作为thread_id
        thread_id = f"{conversation_id}_user"

        # 存储对话ID和thread_id的映射
        self.conversation_map[conversation_id] = thread_id

        return thread_id, conversation_id

    async def get_conversation(self, conversation_id: str):
        """获取特定对话的信息"""
        if not conversation_id or not isinstance(conversation_id, str) or conversation_id.strip() == "":
            return None

        if conversation_id in self.conversation_map:
            return {
                "conversationId": conversation_id,
                "threadId": self.conversation_map[conversation_id]
            }
        else:
            return None

    async def list_conversations(self):
        """
        列出所有对话
        """
        return [
            {
                "conversationId": conv_id,
                "threadId": thread_id
            }
            for conv_id, thread_id in self.conversation_map.items()
        ]

    async def delete_conversation(self, conversation_id: str):
        """
        删除指定的对话

        参数：
            conversation_id (str): 要删除的对话ID

        返回：
            bool: 是否删除成功
        """
        if not conversation_id:
            raise ValueError("对话ID不能为空")

        if not isinstance(conversation_id, str):
            raise ValueError("对话ID必须是字符串类型")

        if conversation_id.strip() == "":
            raise ValueError("对话ID不能为空字符串")

        if conversation_id in self.conversation_map:
            thread_id = self.conversation_map[conversation_id]
            self.data_insight_team.supervisor_agent.checkpointer.delete_thread(thread_id=thread_id)
            del self.conversation_map[conversation_id]
            return True
        else:
            return False

    async def clear_chat_history(self, conversation_id: str):
        """
        清空指定对话的聊天记录

        参数：
            conversation_id (str): 要清空的对话ID

        返回：
            bool: 是否清空成功
        """
        if not conversation_id:
            raise ValueError("对话ID不能为空")

        if not isinstance(conversation_id, str):
            raise ValueError("对话ID必须是字符串类型")

        if conversation_id.strip() == "":
            raise ValueError("对话ID不能为空字符串")

        if conversation_id not in self.conversation_map:
            raise ValueError(f"对话 {conversation_id} 不存在")

        # 这里可以添加实际的清空聊天记录逻辑
        # 由于聊天记录可能由DataInsightTeam管理，我们将简单地返回成功
        thread_id = self.conversation_map[conversation_id]
        self.data_insight_team.supervisor_agent.checkpointer.delete_thread(thread_id=thread_id)

        return True

    async def init_agent(self):
        """初始化Agent"""
        self.data_insight_team = await get_team()

    async def astream(self, content, conversation_id):
        # 获取对话信息
        conversation_map = await self.get_conversation(conversation_id)

        if not conversation_map:
            raise HTTPException(status_code=404, detail=f"对话ID '{conversation_id}' 不存在")

        thread_id = conversation_map['threadId']

        if not self.data_insight_team:
            raise HTTPException(status_code=500, detail="Agent尚未初始化")

        async for message in self.data_insight_team.astream_chat_ui(content=content, thread_id=thread_id):
            yield message

    async def get_history(self, conversation_id: str = "1") -> list[dict[str, Any]]:
        """Get chat history from state."""
        if not conversation_id:
            raise ValueError("对话ID不能为空")

        if not isinstance(conversation_id, str):
            raise ValueError("对话ID必须是字符串类型")

        if conversation_id.strip() == "":
            raise ValueError("对话ID不能为空字符串")

        conversation_map = await self.get_conversation(conversation_id)
        if not conversation_map:
            return []

        thread_id = conversation_map['threadId']

        if not agent_manager.data_insight_team:
            raise RuntimeError("Agent尚未初始化")

        try:
            latest_state = await self.data_insight_team.supervisor_agent.aget_state(
                {"configurable": {"thread_id": thread_id}}, subgraphs=True)
        except Exception as e:
            raise RuntimeError(f"获取对话状态失败: {str(e)}")

        if not latest_state.values:
            return []

        try:
            messages = latest_state.values['messages']
            history_messages = []

            current_msg = {
                'source': 'user' if messages[0].type == 'human' else 'assistant',
                'content': messages[0].content
            }

            for message in messages[1:]:
                # 判断当前消息类型是否与上一条一致
                current_type = 'human' if current_msg['source'] == 'user' else 'ai'
                if message.type == current_type and message.type != 'human':
                    # 同类型：合并内容（换行分隔）
                    current_msg['content'] += '\n' + message.content
                else:
                    # 不同类型：保存当前消息，初始化新消息
                    history_messages.append(current_msg)
                    current_msg = {
                        'source': 'user' if message.type == 'human' else 'assistant',
                        'content': message.content
                    }

            # 保存最后一条消息
            history_messages.append(current_msg)
            return history_messages
        except (KeyError, IndexError, AttributeError) as e:
            raise RuntimeError(f"解析对话历史失败: {str(e)}")


# -------------------------- 文件存储管理 --------------------------

# 文件保存的根目录
base_path = os.path.join(os.path.dirname(__file__), "uploaded_files")
os.makedirs(base_path, exist_ok=True)

# 模拟文件存储系统
file_storage = {
    "所有文件": [],
    "分析结果": []
}


async def save_file(filename: str, content: bytes, folder_name: str = "所有文件"):
    """
    保存文件到指定文件夹

    参数：
        filename (str): 文件名
        content (bytes): 文件内容
        folder_name (str): 目标文件夹名称

    返回：
        dict: 文件信息
    """
    # 保存文件到磁盘
    file_path = os.path.join(base_path, folder_name, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(content)

    # 保存到内存
    if folder_name not in file_storage:
        file_storage[folder_name] = []

    # 检查文件是否已存在
    for file in file_storage[folder_name]:
        if file["name"] == filename:
            # 更新文件内容
            # 检测文件类型
            file_type = filename.split('.')[-1].lower() if '.' in filename else ""

            # 对于数据文件，使用pandas读取为DataFrame
            file_content = content
            if file_type in ['xlsx', 'xls', 'csv']:
                try:
                    if file_type == 'csv':
                        # 读取CSV文件
                        file_content = pd.read_csv(io.BytesIO(content))
                    else:
                        # 读取Excel文件
                        file_content = pd.read_excel(io.BytesIO(content))
                except Exception as e:
                    logging.error(f"读取数据文件失败: {e}")
                    # 如果读取失败，保持原始的bytes内容
                    file_content = content

            file["content"] = file_content.head(10)
            file["size"] = len(content)
            return file

    # 检测文件类型
    file_type = filename.split('.')[-1].lower() if '.' in filename else ""

    # 对于数据文件，使用pandas读取为DataFrame
    file_content = content
    if file_type in ['xlsx', 'xls', 'csv']:
        try:
            if file_type == 'csv':
                # 读取CSV文件
                file_content = pd.read_csv(io.BytesIO(content))
            else:
                # 读取Excel文件
                file_content = pd.read_excel(io.BytesIO(content))
        except Exception as e:
            logging.error(f"读取数据文件失败: {e}")
            # 如果读取失败，保持原始的bytes内容
            file_content = content

    # 创建新文件
    file_info = {
        "name": filename,
        "content": file_content,
        "size": len(content),
        "createTime": "2026-01-12 10:30:00",  # 模拟创建时间
        "type": file_type,  # 提取文件类型
        "fileUrl": file_path
    }

    file_storage[folder_name].append(file_info)

    # 创建不包含content字段的副本返回，避免JSON序列化错误
    file_info_without_content = file_info.copy()
    if 'content' in file_info_without_content:
        del file_info_without_content['content']
    return file_info_without_content


async def get_files_from_folder(folder_name: str):
    """
    获取指定文件夹下的文件列表

    参数：
        folder_name (str): 文件夹名称

    返回：
        list: 文件列表
    """
    if folder_name not in file_storage:
        return []

    # 返回不包含文件内容的文件信息
    return [
        {
            "name": file["name"],
            "size": file["size"],
            "createTime": file["createTime"],
            "type": file["type"]
        }
        for file in file_storage[folder_name]
    ]


async def _get_file_bytes(filename: str, folder_name: str = "所有文件"):
    """
    获取指定文件的原始字节内容，无论文件类型

    参数：
        filename (str): 文件名
        folder_name (str): 文件夹名称

    返回：
        bytes: 文件内容

    异常：
        ValueError: 如果文件不存在
    """
    if folder_name not in file_storage:
        raise ValueError(f"文件夹 '{folder_name}' 不存在")

    for file in file_storage[folder_name]:
        if file["name"] == filename:
            # 如果内容是pandas DataFrame类型，将其转换回bytes
            if isinstance(file["content"], pd.DataFrame):
                file_type = filename.split('.')[-1].lower() if '.' in filename else ""
                buffer = io.BytesIO()

                if file_type == 'csv':
                    # 转换为CSV格式
                    file["content"].to_csv(buffer, index=False)
                else:
                    # 转换为Excel格式
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        file["content"].to_excel(writer, sheet_name='Sheet1', index=False)

                buffer.seek(0)
                return buffer.getvalue()

            # 如果是bytes类型，直接返回
            return file["content"]

    raise ValueError(f"文件 '{filename}' 在文件夹 '{folder_name}' 中不存在")


async def _get_file_content(filename: str, folder_name: str = "所有文件"):
    """
    获取指定文件的内容，根据文件类型返回不同格式

    参数：
        filename (str): 文件名
        folder_name (str): 文件夹名称

    返回：
        dict: 包含文件内容和类型的字典，结构为：
            - content: 文件内容
            - contentType: 内容类型（dataframe、text、base64）
            - fileType: 文件类型

    异常：
        ValueError: 如果文件不存在
    """
    if folder_name not in file_storage:
        raise ValueError(f"文件夹 '{folder_name}' 不存在")

    # 查找文件
    file_info = None
    for file in file_storage[folder_name]:
        if file["name"] == filename:
            file_info = file
            break

    if not file_info:
        raise ValueError(f"文件 '{filename}' 在文件夹 '{folder_name}' 中不存在")

    # 检测文件类型
    file_type = filename.split('.')[-1].lower() if '.' in filename else ""

    # 定义不同类型的文件列表
    text_file_types = ['txt', 'json', 'xml', 'html', 'css', 'js', 'py', 'md', 'yaml', 'yml', 'ini', 'log', 'conf',
                       'properties']
    data_file_types = ['xlsx', 'xls', 'csv']
    image_file_types = ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg']

    # 根据文件类型返回不同格式的内容
    if file_type in data_file_types:
        # 数据文件
        content = file_info["content"]

        if isinstance(content, pd.DataFrame):
            return {
                "content": content,
                "contentType": "dataframe",
                "fileType": file_type,
                "columns": list(content.columns),
                "rows": len(content),
                "sampleData": content.head(5)
            }
        else:
            # 如果不是DataFrame类型，返回bytes并标记为base64
            return {
                "content": content,
                "contentType": "base64",
                "fileType": file_type
            }
    elif file_type in text_file_types:
        # 文本文件
        content_bytes = file_info["content"]
        try:
            content = content_bytes.decode('utf-8')
            return {
                "content": content,
                "contentType": "text",
                "fileType": file_type
            }
        except UnicodeDecodeError:
            # 如果解码失败，返回Base64编码
            return {
                "content": content_bytes,
                "contentType": "base64",
                "fileType": file_type
            }
    else:
        # 图片文件和其他二进制文件，返回Base64编码
        content_bytes = file_info["content"]
        return {
            "content": content_bytes,
            "contentType": "base64",
            "fileType": file_type
        }


# -------------------------- 2. FastAPI生命周期管理 --------------------------

agent_manager = AgentManager()  # 实例化Agent管理器


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI生命周期函数：启动时初始化资源，关闭时清理资源
    """
    # 启动阶段：初始化Agent
    await agent_manager.init_agent()
    logging.info("Agent初始化完成")

    yield  # 应用运行中

    # 关闭阶段：清理资源（如关闭Agent连接、释放内存）
    agent_manager.data_insight_team = None
    logging.info("服务已优雅关闭，资源已清理")


async def shutdown(server, loop: asyncio.AbstractEventLoop):
    """
    优雅关闭服务
    """
    logging.info("收到停止信号，开始优雅关闭...")

    # 停止 Uvicorn 服务器
    if server:
        try:
            # 检查服务器是否在运行
            if hasattr(server, 'should_exit'):
                server.should_exit = True
            # 停止服务器（非阻塞）
            await server.shutdown()
        except Exception as e:
            logging.error(f"关闭服务器时发生错误: {e}")

    # 取消所有未完成的任务
    logging.info("取消所有未完成的任务...")
    tasks_to_cancel = [task for task in asyncio.all_tasks(loop) if task is not asyncio.current_task()]

    for task in tasks_to_cancel:
        task.cancel()

    # 等待任务被取消
    if tasks_to_cancel:
        await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

    logging.info("所有任务已取消，服务已完全关闭")


# -------------------------- 3. 初始化FastAPI应用 --------------------------
app = FastAPI(lifespan=lifespan)

# 配置跨域（前端请求必备）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境替换为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------- 4. 示例接口（测试Agent是否可用） --------------------------
@app.get("/")
async def root():
    """Serve the chat interface HTML file."""
    return FileResponse("chat_ui.html")


@app.post("/new-conversation")
async def new_conversation():
    """
    创建新的对话会话

    返回：
        dict: 包含新创建的对话ID和线程ID
            - conversationId: 新对话的唯一标识符
            - threadId: 对应的线程ID
            - message: 操作结果消息
    """
    # 使用AgentManager创建新对话
    thread_id, conversation_id = await agent_manager.get_or_create_conversation()
    return {"conversationId": conversation_id, "message": "新对话创建成功", "threadId": thread_id}


@app.get("/history-conversations")
async def history_conversations():
    """
    获取所有历史对话列表

    返回：
        list: 对话列表，每个对话包含:
            - conversationId: 对话的唯一标识符
            - threadId: 对应的线程ID
    """
    # 从AgentManager获取所有历史对话列表
    conversations = await agent_manager.list_conversations()
    # 这里可以添加更多的对话信息，如最后一条消息、创建时间等
    return conversations


@app.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """
    获取指定对话的信息

    参数：
        conversation_id (str): 对话ID

    返回：
        dict: 对话信息，包含:
            - conversationId: 对话的唯一标识符
            - threadId: 对应的线程ID

    异常：
        HTTPException: 如果对话不存在
    """
    try:
        conversation = await agent_manager.get_conversation(conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="对话不存在")
        return conversation
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取对话信息失败: {str(e)}") from e


@app.delete("/conversation/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """
    删除指定的对话

    参数：
        conversation_id (str): 要删除的对话ID

    返回：
        dict: 删除结果信息，包含:
            - success: 是否删除成功
            - message: 操作结果消息
    """
    try:
        # 使用AgentManager删除对话
        success = await agent_manager.delete_conversation(conversation_id)
        if success:
            return {
                "success": True,
                "message": f"对话 {conversation_id} 已成功删除"
            }
        else:
            raise HTTPException(status_code=404, detail=f"对话 {conversation_id} 不存在")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除对话失败: {str(e)}")


@app.get("/files/{folder_name}")
async def get_files(folder_name: str):
    """
    获取指定文件夹中的文件列表

    参数：
        folder_name (str): 文件夹名称

    返回：
        list: 文件列表，每个文件包含:
            - name: 文件名
            - size: 文件大小
            - createTime: 文件创建时间
            - type: 文件类型
    """
    # 使用统一的文件存储函数获取文件列表
    files = await get_files_from_folder(folder_name)
    return files


@app.get("/file/{file_name}")
async def get_file_content(file_name: str, folder_name: str = "所有文件"):
    """
    获取指定文件的内容

    参数：
        file_name (str): 文件名
        folder_name (str): 文件夹名称，默认为"所有文件"

    返回：
        dict: 文件内容信息，包含:
            - fileName: 文件名
            - content: 文件内容
            - fileType: 文件类型
            - contentType: 内容类型（text、dataframe、base64）
    """
    try:
        # 调用统一的文件内容获取函数
        content_info = await _get_file_content(file_name, folder_name)

        # 根据内容类型处理返回结果
        if content_info["contentType"] == "dataframe":
            # 数据文件，将DataFrame转换为JSON字符串
            content = content_info["content"]
            content_json = content.to_json(orient='records', force_ascii=False, default_handler=str)
            return {
                "fileName": file_name,
                "content": content_json,
                "fileType": content_info["fileType"],
                "contentType": "dataframe",
                "columns": content_info["columns"],
                "rows": content_info["rows"],
                "sampleData": content_info["sampleData"].to_dict(orient='records')
            }
        elif content_info["contentType"] == "text":
            # 文本文件，直接返回
            return {
                "fileName": file_name,
                "content": content_info["content"],
                "fileType": content_info["fileType"],
                "contentType": "text"
            }
        elif content_info["contentType"] == "base64":
            # 二进制文件，返回Base64编码
            import base64
            content_bytes = content_info["content"]
            content_base64 = base64.b64encode(content_bytes).decode('utf-8')

            # 图片文件添加imageType字段
            if content_info["fileType"] in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp', 'svg']:
                return {
                    "fileName": file_name,
                    "content": content_base64,
                    "fileType": content_info["fileType"],
                    "contentType": "base64",
                    "imageType": content_info["fileType"]
                }
            else:
                return {
                    "fileName": file_name,
                    "content": content_base64,
                    "fileType": content_info["fileType"],
                    "contentType": "base64"
                }
        else:
            # 未知类型，默认返回Base64编码
            import base64
            content_bytes = content_info["content"]
            content_base64 = base64.b64encode(content_bytes).decode('utf-8')
            return {
                "fileName": file_name,
                "content": content_base64,
                "fileType": content_info["fileType"],
                "contentType": "base64"
            }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取文件失败: {str(e)}")


@app.get("/download/{file_name}")
async def download_file(file_name: str, folder_name: str = "所有文件"):
    """
    下载指定文件

    参数：
        file_name (str): 文件名
        folder_name (str): 文件夹名称，默认为"所有文件"

    返回：
        StreamingResponse: 文件流响应，包含文件内容和下载头
    """
    try:
        # 使用统一的文件存储函数获取文件内容
        content_bytes = await _get_file_bytes(file_name, folder_name)
        file_stream = io.BytesIO(content_bytes)

        # 设置适当的媒体类型
        file_type = file_name.split('.')[-1].lower()
        media_type = f"text/{file_type}" if file_type in ['txt', 'csv'] else "application/octet-stream"

        return StreamingResponse(file_stream, media_type=media_type, headers={
            "Content-Disposition": f"attachment; filename={file_name}"
        })
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件下载失败: {str(e)}")


@app.get("/history")
async def history(conversationId: str = "1") -> list[dict[str, Any]]:
    """
    获取指定对话的历史消息记录

    参数：
        conversationId (str): 对话ID，默认为"1"

    返回：
        list: 历史消息列表，每条消息包含:
            - source: 消息来源（user或assistant）
            - content: 消息内容

    异常：
        HTTPException: 如果获取历史消息失败
    """
    try:
        return await agent_manager.get_history(conversationId)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


class ChatRequest(BaseModel):
    """
    聊天请求模型

    属性：
        content (str): 消息内容
        source (str): 消息来源
        conversationId (str): 对话ID
    """
    content: str  # 对应前端的content字段
    source: str
    conversationId: str


@app.post("/chat")
async def chat(request: ChatRequest) -> StreamingResponse:
    """
    发送聊天消息并获取流式响应

    参数：
        request (ChatRequest): 聊天请求，包含消息内容、来源和对话ID

    返回：
        StreamingResponse: 流式响应，包含助手的回复内容

    异常：
        HTTPException: 如果处理聊天请求失败
    """
    # 验证输入参数
    if not request.content:
        raise HTTPException(status_code=400, detail="消息内容不能为空")

    if not request.conversationId:
        raise HTTPException(status_code=400, detail="对话ID不能为空")

    content = request.content.strip()
    conversation_id = request.conversationId.strip()

    if not content:
        raise HTTPException(status_code=400, detail="消息内容不能为空字符串")

    if not conversation_id:
        raise HTTPException(status_code=400, detail="对话ID不能为空字符串")

    try:
        # 获取流式响应
        return StreamingResponse(
            agent_manager.astream(content, conversation_id),
            media_type="text/plain; charset=utf-8")
    except HTTPException:
        # 重新抛出已处理的HTTP异常
        raise
    except Exception as e:
        # 处理其他异常
        error_message = {
            "type": "error",
            "content": f"聊天处理失败: {str(e)}",
            "source": "system"
        }
        raise HTTPException(status_code=500, detail=error_message) from e


@app.post("/chat-with-files")
async def chat_with_files(
        content: str = Form(default=""),  # 允许空内容
        conversationId: str = Form(...),
        source: str = Form(default="user"),
        files: List[UploadFile] = File(default=[])
) -> StreamingResponse:
    """
    发送聊天消息和文件并获取流式响应

    参数：
        content (str): 消息内容
        conversationId (str): 对话ID
        source (str): 消息来源（默认为"user"）
        files (List[UploadFile]): 要上传的文件列表

    返回：
        StreamingResponse: 流式响应，包含助手的回复内容
    """
    # 验证输入参数
    if not conversationId:
        raise HTTPException(status_code=400, detail="对话ID不能为空")

    conversation_id = conversationId.strip()
    if not conversation_id:
        raise HTTPException(status_code=400, detail="对话ID不能为空字符串")

    try:
        # 1. 处理文件上传（保存到默认文件夹）
        default_folder = "所有文件"  # 使用与/upload接口一致的中文文件夹名
        file_paths = []
        for file in files:
            if file.filename:
                # 读取文件内容
                file_content = await file.read()
                if file_content:
                    # 同时保存到内存存储（与/upload接口保持一致）
                    file_info = await save_file(file.filename, file_content, default_folder)
                    file_paths.append(file_info['fileUrl'])

        if content:
            content = content.strip() + '，文件路径为：' + ','.join(file_paths)
        else:
            content = "" + '文件路径为：' + ','.join(file_paths)
        # 2. 发送聊天消息并获取流式响应
        return StreamingResponse(
            agent_manager.astream(content, conversation_id),
            media_type="text/plain; charset=utf-8"
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"聊天处理失败: {str(e)}") from e


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), folderName: str = Form(default="所有文件")):
    """
    上传文件到指定文件夹

    参数：
        file (UploadFile): 要上传的文件对象
        folderName (str): 目标文件夹名称（默认保存到"所有文件"文件夹）

    返回：
        dict: 上传结果信息，包含:
            - success: 上传是否成功
            - filename: 文件名
            - size: 文件大小（字节）
            - fileUrl: 文件下载URL
            - message: 操作结果消息
    """
    # 验证输入参数
    if not file:
        raise HTTPException(status_code=400, detail="文件不能为空")

    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    try:
        # 获取文件内容
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="文件内容不能为空")

        # 保存文件到指定文件夹
        folder_name = folderName.strip() if folderName else "所有文件"

        file_info = await save_file(file.filename, content, folder_name)

        file_info['success'] = True
        file_info['message'] = '文件上传成功'

        return file_info
    except HTTPException:
        # 重新抛出已处理的HTTP异常
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件上传失败: {str(e)}")


@app.post("/clear-chat-history")
async def clear_chat_history(conversationId: str = Form(...)):
    """
    清空指定对话的聊天记录

    参数：
        conversationId (str): 要清空的对话ID

    返回：
        dict: 清空结果信息，包含:
            - success: 是否清空成功
            - message: 操作结果消息
    """
    try:
        # 使用AgentManager清空聊天记录
        success = await agent_manager.clear_chat_history(conversationId)
        if success:
            return {
                "success": True,
                "message": f"对话 {conversationId} 的聊天记录已成功清空"
            }
        else:
            raise HTTPException(status_code=500, detail=f"清空对话 {conversationId} 的聊天记录失败")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"清空聊天记录失败: {str(e)}")


# -------------------------- 5. 启动配置（规范uvicorn启动） --------------------------
def configure_logging():
    """配置日志（生产环境必备）"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),  # 控制台输出
            logging.FileHandler("app.log")  # 日志文件输出
        ]
    )
    # 调整uvicorn日志级别（避免过多调试日志）
    LOGGING_CONFIG["loggers"]["uvicorn"]["level"] = "INFO"


async def main():
    # 配置日志
    configure_logging()
    # 启动uvicorn（使用uvicorn的异步启动方式）
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=8000,
        log_config=LOGGING_CONFIG  # 应用自定义日志配置
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()  # 异步启动uvicorn（避免阻塞）
    except KeyboardInterrupt:
        # 捕获键盘中断异常，确保程序能够优雅退出
        logging.info("程序已退出")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # 捕获键盘中断异常，确保程序能够优雅退出
        print("程序已退出")
