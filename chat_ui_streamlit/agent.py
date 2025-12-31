import sys



def add_current_dir_to_path(current_dir):
    if current_dir not in sys.path:
        sys.path.append(current_dir)
        print(f"已将当前目录添加到环境变量")
    else:
        print(f"当前目录已在环境变量中")


add_current_dir_to_path('')
from data_insight_agent import DataInsightTeam



class Agent:
    def __init__(self) -> None:
        base_url = ""
        api_key = ""
        model = ""

        self.data_insight_team = DataInsightTeam(
            base_url=base_url,
            api_key=api_key,
            model=model, chat_ui=True
        )
    async def chat(self, prompt: str):
        async for message in self.data_insight_team.run_stream_chat_ui(task=prompt):
            print(message)
            yield message

