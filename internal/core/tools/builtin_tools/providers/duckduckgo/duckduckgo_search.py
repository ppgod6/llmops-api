from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from internal.lib.helper import add_attribute


class DDGInput(BaseModel):
    query: str = Field(description="需要搜索的查询信息")


@add_attribute("args_schema", DDGInput)
def duckduckgo_search(**kwargs) -> BaseTool:
    return DuckDuckGoSearchRun(
        description="一个注重隐私的搜索工具，当你需要搜索时事时可以使用该工具，工具的输入是一个查询语句",
        args_schema=DDGInput,
    )
