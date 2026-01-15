import base64
from typing import Any, Dict, List, Tuple

import cv2
import ffmpeg
import numpy as np
import whisper
from autogen_core import Image as AGImage
from autogen_core.models import (
    ChatCompletionClient,
    UserMessage,
)



def extract_audio(video_path: str, audio_output_path: str) -> str:
    """
    从视频文件中提取音频并将其保存为MP3文件。

    :param video_path: 视频文件的路径。
    :param audio_output_path: 提取的音频文件的保存路径。
    :return: 带有已保存音频文件路径的确认消息。
    """
    (ffmpeg.input(video_path).output(audio_output_path, format="mp3").run(quiet=True, overwrite_output=True))  # type: ignore
    return f"音频已提取并保存至：{audio_output_path}。"





def transcribe_audio_with_timestamps(audio_path: str) -> str:
    """
    将带有时间戳的音频文件转录为文本信息。

    :param audio_path: 音频文件的路径。
    :return: 带时间戳的转录文本信息。
    """
    model = whisper.load_model("medium")  # type: ignore
    result: Dict[str, Any] = model.transcribe(audio_path, task="transcribe", language="zh", verbose=False)  # type: ignore

    segments: List[Dict[str, Any]] = result["segments"]
    transcription_with_timestamps = ""

    for segment in segments:
        start: float = segment["start"]
        end: float = segment["end"]
        text: str = segment["text"]
        transcription_with_timestamps += f"[{start:.2f} - {end:.2f}] {text}\n"

    return transcription_with_timestamps






def get_video_length(video_path: str) -> str:
    """
    以秒为单位返回视频的长度。

    :param video_path: 视频文件的路径。
    :return: 以秒为单位的视频时长。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps
    cap.release()
    timestamps = np.linspace(start=0, stop=duration, num=int(duration / 5)).astype(int)

    return f"视频时长为{duration:.2f}秒，建议抽取时间戳为{'秒,'.join(timestamps.astype(str)) + '秒'}的视频截屏进行分析。"




def save_screenshot(video_path: str, timestamp: float, output_path: str) -> str:
    """
    在指定的时间戳捕获屏幕截图并将其保存到输出路径。

    :param video_path: 视频文件的路径。
    :param timestamp: 以秒为单位的时间戳。
    :param output_path: 保存屏幕截图的路径。文件格式由路径中的扩展名决定。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件{video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps
    frame_number = int(timestamp * fps)
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    # ret, frame = cap.read()
    # if ret:
    #     cv2.imwrite(output_path, frame)
    # else:
    #     raise IOError(f"在{timestamp:.2f}秒截图失败")
    current_frame_number = -1
    ret = True
    save = False
    # 逐帧读取，统计真实帧数
    while ret:
        ret, frame = cap.read()
        if ret:
            current_frame_number += 1
        if ret and current_frame_number == np.minimum(frame_number, int(duration * fps) - 1):
            cv2.imwrite(output_path, frame)
            save = True
            ret = False
    cap.release()
    if save:
        return f'视频截图已保存至{output_path}。'
    else:
        return '视频截图已保存失败。'



async def transcribe_video_screenshot(video_path: str, timestamp: float, model_client: ChatCompletionClient) -> str:
    """
    将指定时间戳捕获的视频截图的内容转录为文本信息。

    :param video_path: 视频文件的路径。
    :param timestamp: 以秒为单位的时间戳。
    :param model_client: ChatCompletionClient的实例。
    :return: 屏幕截图内容的描述。
    """
    screenshots = get_screenshot_at(video_path, [timestamp])
    if not screenshots:
        return "屏幕截图失败。"

    _, frame = screenshots[0]
    # Convert the frame to bytes and then to base64 encoding
    _, buffer = cv2.imencode(".jpg", frame)
    frame_bytes = buffer.tobytes()
    frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")
    screenshot_uri = f"data:image/jpeg;base64,{frame_base64}"

    messages = [
        UserMessage(
            content=[
                f"以下是{timestamp}秒时视频的屏幕截图。请描述一下你在这里所看到的信息。",
                AGImage.from_uri(screenshot_uri),
            ],
            source="tool",
        )
    ]

    result = await model_client.create(messages=messages)
    return str(result.content)




def get_screenshot_at(video_path: str, timestamps: List[float]) -> List[Tuple[float, np.ndarray[Any, Any]]]:
    """
    在指定的时间戳捕获屏幕截图，并将其作为Python对象返回。

    :param video_path: 视频文件的路径。
    :param timestamps: 以秒为单位的时间戳所组成的列表。
    :return: 包含时间戳和相应帧（图像）的元组列表。
            每一帧都是一个NumPy数组（高x宽x通道）。
    """
    screenshots: List[Tuple[float, np.ndarray[Any, Any]]] = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"无法打开视频文件{video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = total_frames / fps

    cap.release()

    for timestamp in timestamps:
        if 0 <= timestamp <= duration:
            frame_number = int(timestamp * fps)
            # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            # ret, frame = cap.read()
            # if ret:
            #     # Append the timestamp and frame to the list
            #     screenshots.append((timestamp, frame))
            # else:
            #     raise IOError(f"在{timestamp:.2f}秒截图失败")
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"无法打开视频文件{video_path}")
            current_frame_number = -1
            ret = True
            # 逐帧读取，统计真实帧数
            while ret:
                ret, frame = cap.read()
                if ret:
                    current_frame_number += 1
                if ret and current_frame_number == np.minimum(frame_number, int(duration * fps) - 1):
                    screenshots.append((timestamp, frame))
                    ret = False

            cap.release()


        else:
            raise ValueError(f"时间戳{timestamp:.2f}秒超出了视频时长[0s, {duration:.2f}s]的范围")


    return screenshots



