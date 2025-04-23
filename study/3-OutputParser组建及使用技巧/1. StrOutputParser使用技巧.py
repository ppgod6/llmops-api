import os

import dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()

prompt = ChatPromptTemplate.from_template("{query}")

llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("OPENAI_API_KEY"))

# 创建字符串输出解析器
parser = StrOutputParser()

content = parser.invoke(llm.invoke(prompt.invoke({"query": "你好，你是？"})))

print(content)
