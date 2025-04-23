import os
from datetime import datetime

import dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()

# 1.编排prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是deepeek开发的聊天机器人，请回答用户的问题，现在的时间是{now}"),
    ("human", "{query}")
]).partial(now=datetime.now())

# 2.创建大语言模型
llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("OPENAI_API_KEY"))

ai_messages = llm.batch([
    prompt.invoke({"query": "你好，你是？"}),
    prompt.invoke({"query": "请讲一个关于程序员的冷笑话"}),
])

for ai_messages in ai_messages:
    print(ai_messages.content)
    print("=================")
