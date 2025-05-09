from abc import ABC, abstractmethod
from typing import Generator

from internal.core.agent.entities.queue_entity import AgentQueueEvent
from langchain_core.messages import AnyMessage

from internal.core.agent.entities.agent_entity import AgentConfig
from .agent_queue_manager import AgentQueueManager


class BaseAgent(ABC):
    """LLMOps项目基础Agent"""
    agent_config: AgentConfig
    agent_queue_manager: AgentQueueManager

    def __init__(
            self,
            agent_config: AgentConfig,
            agent_queue_manager: AgentQueueManager,
    ):
        """构造函数，初始化智能体图结构程序"""
        self.agent_config = agent_config
        self.agent_queue_manager = agent_queue_manager

    @abstractmethod
    def run(
            self,
            query: str,  # 用户提问原始问题
            history: list[AnyMessage] = None,  # 短期记忆
            long_term_memory: str = "",  # 长期记忆
    ) -> Generator[AgentQueueEvent, None, None]:
        """智能体运行函数，传递原始提问query、长短期记忆，并调用智能体生成相应内容"""
        raise NotImplementedError("Agent智能体的run函数未实现")
