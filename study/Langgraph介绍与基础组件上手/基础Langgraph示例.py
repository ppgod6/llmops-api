import os

import dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

dotenv.load_dotenv()
from typing import TypedDict, Annotated, Any
from langchain_deepseek import ChatDeepSeek

llm = ChatDeepSeek(model="deepseek-chat", api_key=os.getenv("OPENAI_API_KEY"))


class State(TypedDict):
    """图结构的状态数据"""
    messages: Annotated[list, add_messages]


def chatbot(state: State, config: dict) -> Any:
    ai_messages = llm.invoke(state["messages"])
    return {"messages": [ai_messages]}


graph_builder = StateGraph(State)
graph_builder.add_node("llm", chatbot)
graph_builder.add_edge(START, "llm")
graph_builder.add_edge("llm", END)

graph = graph_builder.compile()

print(graph.invoke({"messages": [("human", "你好，你是谁，我是马卡，我喜欢打篮球")]}))

