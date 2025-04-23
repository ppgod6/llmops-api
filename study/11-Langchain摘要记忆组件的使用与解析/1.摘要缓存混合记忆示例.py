import os
from typing import List, Dict

import dotenv
import tiktoken
from langchain.memory import ConversationSummaryBufferMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()

# 初始化 DeepSeek 模型
llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("OPENAI_API_KEY"))

# 自定义 Token 计数器 (兼容 cl100k_base)
encoding = tiktoken.get_encoding("cl100k_base")


def calculate_tokens(messages: List[Dict]) -> int:
    """直接计算消息列表的总 token 数"""
    total = 0
    for msg in messages:
        if isinstance(msg, dict):
            content = msg.get("content", "")
        else:  # 处理 Message 对象
            content = msg.content
        total += len(encoding.encode(content))
    return total


# 彻底自定义的 Memory 类
class DeepSeekMemory(ConversationSummaryBufferMemory):
    def _get_num_tokens_from_messages(self, messages: List[Dict]) -> int:
        """覆盖内部方法，完全使用自定义 token 计数"""
        return calculate_tokens(messages)

    def count_tokens(self, messages: List[Dict]) -> int:
        """覆盖 count_tokens 方法"""
        return self._get_num_tokens_from_messages(messages)


# 初始化 Memory
memory = DeepSeekMemory(
    llm=llm,
    max_token_limit=300,
    return_messages=True,
    input_key="query",
    human_prefix="Human",
    ai_prefix="AI"
)

# 提示模板
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是由DeepSeek开发的智能助手，请用中文回答用户问题"),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{query}"),
])

# 构建链
chain = (
        RunnablePassthrough.assign(
            history=RunnableLambda(lambda _: memory.load_memory_variables({})["history"])
        )
        | prompt
        | llm
        | StrOutputParser()
)

# 对话循环
while True:
    query = input("Human: ")
    if query == "q":
        break

    # 流式响应
    response = chain.stream({"query": query})
    print("AI: ", end="", flush=True)
    output = ""
    for chunk in response:
        output += chunk
        print(chunk, end="", flush=True)

    # 手动保存上下文并检查 Token 限制
    memory.save_context({"query": query}, {"output": output})
    current_tokens = calculate_tokens(memory.buffer)
    if current_tokens > memory.max_token_limit:
        memory.prune()  # 触发自动修剪

    print(f"\n当前 Token 数: {current_tokens}")
