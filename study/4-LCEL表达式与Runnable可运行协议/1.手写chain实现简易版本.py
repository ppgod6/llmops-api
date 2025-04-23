import os
from typing import Any

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()

# 1.构建组件

prompt = ChatPromptTemplate.from_template("{query}")
llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("OPENAI_API_KEY"))
parser = StrOutputParser()


class Chain:
    steps: list = []

    def __init__(self, steps: list):
        self.steps = steps

    def invoke(self, input: Any) -> Any:
        for step in self.steps:
            input = step.invoke(input)
            print("步骤；", step)
            print("输出；", input)
            print("=============")

        return input


chain = Chain([prompt, llm, parser])

print(chain.invoke({"query": "你好，你是？"}))
