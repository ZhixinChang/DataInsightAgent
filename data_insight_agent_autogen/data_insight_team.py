from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import ExternalTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.tools import TeamTool
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily
from autogen_ext.models.openai import OpenAIChatCompletionClient
from pydantic import BaseModel

from .analytical_tools import get_from_user, data_describe, correlation_analysis, get_from_user_chat_ui, \
    get_summary_info, save_summary
from .console_chat_ui import console_chat_ui
from .video_tools import get_video_length, save_screenshot, transcribe_video_screenshot


class AgentResponse(BaseModel):
    response: str
    result_path: str


class DataInsightTeam:
    def __init__(self, base_url, api_key, model, chat_ui=False):
        self.model_client = OpenAIChatCompletionClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0,
            parallel_tool_calls=False,
            model_info={
                "vision": True,
                "function_calling": True,
                "json_output": False,
                "family": ModelFamily.R1,
                "structured_output": True,
                'multiple_system_messages': False
            },
        )
        self.model_client_parallel_tool_calls = OpenAIChatCompletionClient(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=0,
            parallel_tool_calls=True,
            model_info={
                "vision": True,
                "function_calling": True,
                "json_output": False,
                "family": ModelFamily.R1,
                "structured_output": True,
                'multiple_system_messages': False
            },
        )
        if chat_ui:
            vs_get_from_user = get_from_user_chat_ui
        else:
            vs_get_from_user = get_from_user

        # 数据分析团队
        data_analyst_agent = AssistantAgent(name='Data_analyst', model_client=self.model_client,
                                            system_message='你是一个数据分析专家，请使用工具解决分析问题，先使用data_describe获取数据集相关信息，再使用其他工具进行分析。当使用工具所需信息不足或工具使用报错时请使用get_from_user向用户获取信息补充，当分析完成时请输出工具返回的结果并在输出的结尾回复<DATA_ANALYST_TERMINATE>。',
                                            tools=[vs_get_from_user, data_describe, correlation_analysis],
                                            reflect_on_tool_use=False
                                            )

        user_termination = TextMentionTermination("USER_TERMINATE")
        data_analyst_termination = TextMentionTermination("DATA_ANALYST_TERMINATE")
        self.external_termination = ExternalTermination()

        self.data_analysis_team = RoundRobinGroupChat(participants=[data_analyst_agent],
                                                      termination_condition=user_termination | data_analyst_termination | self.external_termination)

        data_analysis_team_tool = TeamTool(
            team=self.data_analysis_team,
            name="data_analysis_team",
            description="数据分析团队工具，负责完成指标相关性的分析，task信息至少需要包含数据集路径、结果保存路径和分析需求的信息",
            return_value_as_last_message=True,
        )

        # 视频分析团队
        video_analyst_agent = AssistantAgent(name='Video_analyst', model_client=self.model_client_parallel_tool_calls,
                                             system_message='你是一个视频分析专家，请使用工具解决分析问题，先使用get_video_length获取视频相关信息，再使用其他工具进行分析。当使用工具所需信息不足或工具使用报错时请使用get_from_user向用户获取信息补充，当分析完成时请输出工具返回的结果并在输出的结尾回复<VIDEO_ANALYST_TERMINATE>。',
                                             tools=[vs_get_from_user, get_video_length, save_screenshot,
                                                    self.vs_transribe_video_screenshot],
                                             reflect_on_tool_use=False
                                             )
        video_analyst_termination = TextMentionTermination("VIDEO_ANALYST_TERMINATE")
        self.video_analysis_team = RoundRobinGroupChat(participants=[video_analyst_agent],
                                                       termination_condition=user_termination | video_analyst_termination)

        video_analysis_team_tool = TeamTool(
            team=self.video_analysis_team,
            name="video_analysis_team",
            description="视频分析团队工具，负责完成视频内容分析",
            return_value_as_last_message=True,
        )

        # 总结人员
        summary_agent = AssistantAgent(name='Summarizer', model_client=self.model_client,
                                       system_message='你是一个结论总结的专家，请先使用get_summary_info获取结论总结所需要的信息，再以Markdown的格式进行总结，并使用save_summary输出保存，不要额外补充没获取到的信息。当使用工具所需信息不足时请说明需要从用户获取额外信息的原因。无论最终是否完成结论总结都需要在输出的结尾回复<SUMMARIZER_TERMINATE>',
                                       tools=[get_summary_info, save_summary]
                                       )
        summarizer_termination = TextMentionTermination("SUMMARIZER_TERMINATE")
        self.summarizer_team = RoundRobinGroupChat(participants=[summary_agent],
                                                   termination_condition=user_termination | summarizer_termination | self.external_termination)

        summarizer_team_tool = TeamTool(
            team=self.summarizer_team,
            name="summarizer_team",
            description="结论总结团队工具，task信息至少需要包含用户需求、交付结论和结果保存路径的信息，团队负责将这些信息进行总结整理",
            return_value_as_last_message=True,
        )

        # 团队主管
        manager_agent = AssistantAgent(name='Manager', model_client=self.model_client,
                                       system_message='你是一个数据团队的管理者，请结合用户需求使用合适的团队工具解决分析问题，请客观地传达任务信息，不要额外补充用户没提到的需求。当分析完成时请输出工具返回的结果并在输出的结尾回复<MANAGER_TERMINATE>。',
                                       tools=[data_analysis_team_tool, video_analysis_team_tool, summarizer_team_tool]
                                       )
        manager_termination = TextMentionTermination("MANAGER_TERMINATE")
        self.manager_team = RoundRobinGroupChat(participants=[manager_agent],
                                                termination_condition=user_termination | manager_termination | self.external_termination)

        # 审核人员
        reviewer_agent = AssistantAgent(name='Reviewer', model_client=self.model_client,
                                        system_message='''你是需求交付的审核专家，请结合用户需求和Manager的回答评估用户的问题是否得到解决。当用户需求没被Manager解决时，请指出未解决的部分，当用户需求确认已经被Manager解决时请直接回复<REVIEWER_TERMINATE>，不需要其他内容。'''
                                        )
        reviewer_termination = TextMentionTermination("REVIEWER_TERMINATE")

        # 数据洞察团队
        self.data_insight_team = RoundRobinGroupChat(participants=[self.manager_team, reviewer_agent],
                                                     termination_condition=user_termination | reviewer_termination,
                                                     max_turns=5)

    async def run_stream(self, task):

        while True:
            self.result = await Console(self.data_insight_team.run_stream(task=task), output_stats=True)
            task = input("请输入你的反馈（输入“停止”结束任务）：")
            if task.lower().strip() == "停止":
                break
        return self.result

    async def run_stream_chat_ui(self, task):

        async for message in console_chat_ui(self.data_insight_team.run_stream(task=task, output_task_messages=False),
                                             output_stats=True):
            yield message

    async def close(self):
        await self.model_client.close()

    async def vs_transribe_video_screenshot(self, video_path: str, timestamp: float) -> str:
        """
        将指定时间戳捕获的视频截图的内容转录为文本信息。

        Args:
            video_path (str): 视频文件的路径。
            timestamp (float): 以秒为单位的时间戳。

        Returns:
            str: 屏幕截图内容的描述。
        """
        return await transcribe_video_screenshot(video_path, timestamp, self.model_client)
