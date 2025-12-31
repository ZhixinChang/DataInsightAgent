import asyncio
import os
import re

import streamlit as st

from agent import Agent


async def chat_stream(prompt, response_placeholder):
    response = ''
    images = []
    files = []
    async for message in st.session_state["agent"].chat(prompt):
        if isinstance(message, list):
            message_path = ','.join(message)
            image_paths_list = re.findall(pattern=r'IMAGE_PATHS<(.*)>IMAGE_PATHS', string=message_path)
            if image_paths_list:
                for image_paths in image_paths_list:
                    image_path_list = image_paths.split(',')
                    images += image_path_list
            file_paths_list = re.findall(pattern=r'FILE_PATHS<(.*)>FILE_PATHS', string=message_path)
            if file_paths_list:
                for file_paths in file_paths_list:
                    file_path_list = file_paths.split(',')
                    files += file_path_list

        else:
            response_placeholder.markdown(response + "▌", unsafe_allow_html=True)
            for char in list(message):
                response += char
                # 关键：更新占位符内容，实现实时刷新
                response_placeholder.markdown(response + "▌", unsafe_allow_html=True)  # 加光标效果，更逼真
                await asyncio.sleep(0.005)

    # 流式结束：移除光标，固定最终内容
    response_placeholder.markdown(response, unsafe_allow_html=True)
    for image in images:
        st.image(image, caption="", width=400)
    for file in files:
        # 读取md文件内容（指定UTF-8编码，避免中文乱码）
        try:
            with open(file, "r", encoding="utf-8") as f:
                md_text = f.read()
        except FileNotFoundError:
            st.error(f"文件未找到：{file}")
            md_text = ""

        st.markdown(os.path.basename(file) + '的具体结论如下：👇', unsafe_allow_html=True)
        st.markdown(md_text, unsafe_allow_html=True)

    return response, images, files


def main() -> None:
    st.set_page_config(page_title="Data Insight Team", page_icon="🤖", layout="wide")
    st.markdown("<center><h1>Data Insight Team 🤖</h1></center>", unsafe_allow_html=True)

    # adding agent object to session state to persist across sessions
    # streamlit reruns the script on every user interaction
    if "agent" not in st.session_state:
        st.session_state["agent"] = Agent()

    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # 初始化时创建全局事件循环
    if "loop" not in st.session_state:
        st.session_state.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(st.session_state.loop)

    # displaying chat history messages
    for message in st.session_state["messages"]:
        if message["role"] == 'user':
            with st.chat_message(message["role"], width='content'):
                st.markdown(message["content"], unsafe_allow_html=True)
        if message["role"] == 'assistant':
            with st.chat_message(message["role"], width='content'):
                with st.expander("点击查看/收起聊天记录", expanded=False):
                    st.markdown(message["content"], unsafe_allow_html=True)
                    if 'image' in message:
                        for image in message['image']:
                            st.image(image, caption="", width=400)
                    if 'file' in message:
                        for file in message['file']:
                            try:
                                with open(file, "r", encoding="utf-8") as f:
                                    md_text = f.read()
                            except FileNotFoundError:
                                st.error(f"文件未找到：{file}")
                                md_text = ""

                            st.markdown(os.path.basename(file) + '的具体结论如下：👇', unsafe_allow_html=True)
                            st.markdown(md_text, unsafe_allow_html=True)

    prompt = st.chat_input("Type a message...")
    if prompt:
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user", width='content'):
            st.markdown(prompt)

        with st.chat_message('assistant', width='content'):
            response_placeholder = st.empty()

            # 后续操作使用该循环，而非asyncio.run()
            response, images, files = st.session_state.loop.run_until_complete(
                chat_stream(prompt, response_placeholder)
            )

            st.session_state["messages"].append(
                {"role": "assistant", "content": response, 'image': images, 'file': files})


if __name__ == "__main__":
    main()
