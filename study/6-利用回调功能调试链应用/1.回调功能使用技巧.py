import os
import time
from typing import Any, Optional
from uuid import UUID

import dotenv
from langchain_core.callbacks import StdOutCallbackHandler, BaseCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import LLMResult
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_deepseek import ChatDeepSeek

dotenv.load_dotenv()


class LLMOpsCallbackHandler(BaseCallbackHandler):
    """自定义LLMOps回调处理器"""
    start_at: float = 0

    def on_chat_model_start(
            self,
            serialized: dict[str, Any],
            messages: list[list[BaseMessage]],
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            tags: Optional[list[str]] = None,
            metadata: Optional[dict[str, Any]] = None,
            **kwargs: Any,
    ) -> Any:
        print("聊天模型开始执行了")
        print("serialized:", serialized)
        print("messages:", messages)
        self.start_at = time.time()

    def on_llm_end(
            self,
            response: LLMResult,
            *,
            run_id: UUID,
            parent_run_id: Optional[UUID] = None,
            **kwargs: Any,
    ) -> Any:
        end_at: float = time.time()
        print("完整输出：", response)
        print("程序消耗：", end_at - self.start_at)


# 1.构建组件

prompt = ChatPromptTemplate.from_template("{query}")
llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("OPENAI_API_KEY"))
parser = StrOutputParser()

# 2.创建链
chain = {"query": RunnablePassthrough()} | prompt | llm | parser

resp = chain.stream(
    "你好，你是？",
    config={"callbacks": [StdOutCallbackHandler(), LLMOpsCallbackHandler()]})

for chunk in resp:
    pass
