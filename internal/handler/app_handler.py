import json
import os
import uuid
from dataclasses import dataclass
from operator import itemgetter
from queue import Queue
from threading import Thread
from typing import Any, Dict, Literal, Generator
from uuid import UUID

from injector import inject
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.documents import Document
from langchain_core.memory import BaseMemory
from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableConfig
from langchain_core.tracers import Run
from langchain_deepseek import ChatDeepSeek
from langgraph.constants import END
from langgraph.graph import MessagesState, StateGraph

from internal.core.tools.builtin_tools.providers import BuiltinProviderManager
from internal.schema.app_schema import CompletionReq
from internal.service import AppService, VectorDatabaseService, ConversationService
from pkg.response import success_json, validate_error_json, success_message, compact_generate_response


@inject
@dataclass
class AppHandler:
    """应用控制器"""
    app_service: AppService
    builtin_provider_manager: BuiltinProviderManager
    vector_database_service: VectorDatabaseService
    conversation_service: ConversationService

    # vector_database_service:

    def create_app(self):
        """调用服务创建新的APP记录"""
        app = self.app_service.create_app()
        return success_message(f"应用已经成功创建，id为{app.id}")

    def get_app(self, id: uuid.UUID):
        app = self.app_service.get_app(id)
        return success_message(f"应用已经成功获取，名字是{app.name}")

    def update_app(self, id: uuid.UUID):
        app = self.app_service.update_app(id)
        return success_message(f"应用已经成功修改，修改的名字是：{app.name}")

    def delete_app(self, id: uuid.UUID):
        app = self.app_service.delete_app(id)
        return success_message(f"应用已经成功删除，id为：{app.id}")

    @classmethod
    def _load_memory_variables(cls, input: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:

        configurable = config.get("configurable", {})
        configurable_memory = configurable.get("memory", None)
        if configurable_memory is not None and isinstance(configurable_memory, BaseMemory):
            return configurable_memory.load_memory_variables(input)
        return {"history": []}

    @classmethod
    def _save_context(cls, run_obj: Run, config: RunnableConfig) -> None:
        configurable = config.get("configurable", {})
        configurable_memory = configurable.get("memory", None)
        if configurable_memory is not None and isinstance(configurable_memory, BaseMemory):
            configurable_memory.save_context(run_obj.inputs, run_obj.outputs)

    def debug(self, app_id: UUID):
        """聊天接口"""
        # 1.提取从接口中获取的输入
        req = CompletionReq()
        if not req.validate():
            return validate_error_json(req.errors)

        q = Queue()
        query = req.query.data

        def graph_app() -> None:
            tools = [
                self.builtin_provider_manager.get_tool("google", "google_serper")(),
                self.builtin_provider_manager.get_tool("gaode", "gaode_weather")(),
                self.builtin_provider_manager.get_tool("dalle", "dalle3")(),
            ]

            def chatbot(state: MessagesState) -> MessagesState:
                llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("OPENAI_API_KEY"),
                                   temperature=0.7).bind_tools(tools)

                is_first_chunk = True
                is_tool_call = False
                gathered = None
                id = str(uuid.uuid4())
                for chunk in llm.stream(state["messages"]):
                    if is_first_chunk and chunk.content == "" and not chunk.tool_calls:
                        continue
                    if is_first_chunk:
                        gathered = chunk
                        is_first_chunk = False
                    else:
                        gathered += chunk
                    if chunk.tool_calls or is_tool_call:
                        is_tool_call = True
                        q.put({
                            "id": id,
                            "event": "agent_thought",
                            "data": json.dumps(chunk.tool_call_chunks),
                        })
                    else:
                        q.put({
                            "id": id,
                            "event": "agent_message",
                            "data": chunk.content,
                        })

                return {"messages": [gathered]}

            def tool_executor(state: MessagesState) -> MessagesState:
                """工具执行节点"""
                # 3.3.1 提取数据状态中的tool_calls
                tool_calls = state["messages"][-1].tool_calls

                # 3.3.2 将工具列表转换成字典便于使用
                tools_by_name = {tool.name: tool for tool in tools}

                # 3.3.3 执行工具并得到对应的结果
                messages = []
                for tool_call in tool_calls:
                    id = str(uuid.uuid4())
                    tool = tools_by_name[tool_call["name"]]
                    tool_result = tool.invoke(tool_call["args"])
                    messages.append(ToolMessage(
                        tool_call_id=tool_call["id"],
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                    ))
                    q.put({
                        "id": id,
                        "event": "agent_action",
                        "data": json.dumps(tool_result),
                    })

                return {"messages": messages}

            def route(state: MessagesState) -> Literal["tool_executor", "__end__"]:
                """定义路由节点，用于确认下一步步骤"""
                ai_message = state["messages"][-1]
                if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
                    return "tool_executor"
                return END

            graph_builder = StateGraph(MessagesState)

            graph_builder.add_node("llm", chatbot)
            graph_builder.add_node("tool_executor", tool_executor)

            graph_builder.set_entry_point("llm")
            graph_builder.add_conditional_edges("llm", route)
            graph_builder.add_edge("tool_executor", "llm")

            graph = graph_builder.compile()

            result = graph.invoke({"messages": [("human", query)]})
            print("最终结果: ", result)
            q.put(None)

        def stream_event_response() -> Generator:
            """流式事件输出响应"""
            # 1.从队列中获取数据并使用yield抛出
            while True:
                item = q.get()
                if item is None:
                    break
                # 2.使用yield关键字返回对应的数据
                yield f"event: {item.get('event')}\ndata: {json.dumps(item)}\n\n"
                q.task_done()

        t = Thread(target=graph_app)
        t.start()

        return compact_generate_response(stream_event_response())

    def _debug(self, app_id: UUID):
        """聊天接口"""
        # 1.提取从接口中获取的输入
        req = CompletionReq()
        if not req.validate():
            return validate_error_json(req.errors)

        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个强大的聊天机器人，能根据用户的提问回复对应的问题"),
            MessagesPlaceholder("history"),
            ("human", "{query}"),
        ])

        memory = ConversationBufferWindowMemory(
            k=3,
            memory_key="history",
            input_key="query",
            output_key="output",
            return_messages=True,
            chat_memory=FileChatMessageHistory("./storage/memory/chat_history.txt"),
        )

        llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("OPENAI_API_KEY"))
        retriever = self.vector_database_service.get_retriever() | self.vector_database_service.combine_documents
        chain = (RunnablePassthrough.assign(
            history=RunnableLambda(self._load_memory_variables) | itemgetter("history"),
            context=itemgetter("query") | retriever
        ) | prompt | llm | StrOutputParser()).with_listeners(on_end=self._save_context)

        # 5.调用链生成内容
        chain_input = {"query": req.query.data}
        content = chain.invoke(chain_input, config={"configurable": {"memory": memory}})

        return success_json({"content": content})

    @classmethod
    def _combine_documents(cls, documents: list[Document]) -> str:
        """将传入的文档列表合并成字符串"""
        return "\n\n".join([document.page_content for document in documents])

    def ping(self):
        human_message = "我是马卡"
        ai_message = "你是马卡，你喜欢打篮球"
        summary = self.conversation_service.summary(human_message, ai_message)
        return success_json({"summary": summary})
